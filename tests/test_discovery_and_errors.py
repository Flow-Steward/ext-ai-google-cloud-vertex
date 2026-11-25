"""Google's error answers map to Flow Steward's categories; Discover lists only chat
models the project can call in its location, using free countTokens checks.

Bodies are the ones Google returned on a real project (2026-11-25).
"""

from __future__ import annotations

import json
import time
from typing import Any

import pytest
import requests

import main
import runtime

BILLING_DISABLED = {
    "error": {
        "code": 403,
        "message": "This API method requires billing to be enabled. Please enable billing ...",
        "status": "PERMISSION_DENIED",
        "details": [
            {"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": "BILLING_DISABLED"}
        ],
    }
}
FREE_TRIAL = {
    "error": {
        "code": 403,
        "message": "Access requires an eligible paid billing account in a supported region, "
        "subject to our terms and policies.",
        "status": "PERMISSION_DENIED",
    }
}
SERVICE_DISABLED = {
    "error": {
        "code": 403,
        "message": "Vertex AI API has not been used in project 123 before or it is disabled.",
        "status": "PERMISSION_DENIED",
        "details": [{"reason": "SERVICE_DISABLED"}],
    }
}
IAM_DENIED = {
    "error": {
        "code": 403,
        "message": "Permission 'aiplatform.endpoints.predict' denied on resource.",
        "status": "PERMISSION_DENIED",
        "details": [{"reason": "IAM_PERMISSION_DENIED"}],
    }
}
NOT_FOUND = {
    "error": {
        "code": 404,
        "message": "Publisher model `projects/p/locations/us-central1/publishers/google/models/"
        "gemini-2.5-flash` was not found or your project does not have access to it.",
        "status": "NOT_FOUND",
    }
}
INVALID_ARGUMENT = {"error": {"code": 400, "message": "...", "status": "INVALID_ARGUMENT"}}


def body(answer: dict[str, Any]) -> bytes:
    return json.dumps(answer).encode()


# -- table A: one category per Google answer ----------------------------------------------


@pytest.mark.parametrize(
    ("status", "answer", "category"),
    [
        (403, BILLING_DISABLED, "billing_quota_exceeded"),
        (403, FREE_TRIAL, "billing_quota_exceeded"),
        (403, SERVICE_DISABLED, "authentication_error"),
        (403, IAM_DENIED, "authentication_error"),
        (403, {}, "authentication_error"),
        (401, {}, "authentication_error"),
        (404, NOT_FOUND, "model_unavailable"),
        (429, {"error": {"status": "RESOURCE_EXHAUSTED"}}, "rate_limited"),
        (500, {}, "upstream_unavailable"),
        (503, {}, "upstream_unavailable"),
        (409, {}, "invalid_response"),
    ],
)
def test_each_google_answer_has_its_category(status, answer, category):
    assert runtime.classify_error(status, body(answer)) == category


def test_a_bad_request_is_the_callers_category():
    raw = body(INVALID_ARGUMENT)
    assert runtime.classify_error(400, raw) == "invalid_response"
    assert runtime.classify_error(400, raw, bad_request="capability_mismatch") == (
        "capability_mismatch"
    )


@pytest.mark.parametrize("raw", [b"", b"not json", b"[]", b'{"error": "x"}', b"\xff\xfe"])
def test_an_unreadable_403_body_is_access_not_a_crash(raw):
    assert runtime.classify_error(403, raw) == "authentication_error"


class _Response:
    def __init__(self, status: int, payload: bytes) -> None:
        self.status_code = status
        self._payload = payload

    def iter_content(self, size: int):
        for start in range(0, len(self._payload), size):
            yield self._payload[start : start + size]


def _session_answering(status: int, payload: bytes, calls: list | None = None):
    class Session:
        trust_env = True

        def request(self, *args, **kwargs):
            if calls is not None:
                calls.append(kwargs)
            return _Response(status, payload)

        def close(self):
            pass

    return Session


class _Credentials:
    def before_request(self, *args):
        pass


def test_request_json_reports_the_category_and_status_without_google_text(monkeypatch):
    monkeypatch.setattr(requests, "Session", _session_answering(403, body(FREE_TRIAL)))
    with pytest.raises(runtime.VertexHttpError) as exc:
        runtime.request_json("https://aiplatform.googleapis.com/v1/x", _Credentials(), body={})
    assert str(exc.value) == "billing_quota_exceeded"
    assert exc.value.status == 403
    assert "eligible" not in str(exc.value)


def test_request_json_reads_only_a_bounded_error_body(monkeypatch):
    huge = b'{"error":{"message":"' + b"x" * (runtime.MAX_ERROR_BODY_BYTES * 4) + b'"}}'
    monkeypatch.setattr(requests, "Session", _session_answering(403, huge))
    with pytest.raises(runtime.VertexHttpError, match="authentication_error"):
        runtime.request_json("https://aiplatform.googleapis.com/v1/x", _Credentials())


def test_chat_maps_a_400_to_capability_mismatch(monkeypatch):
    monkeypatch.setattr(
        runtime, "connection", lambda request: ("sample-project", "global", _Credentials())
    )
    monkeypatch.setattr(requests, "Session", _session_answering(400, body(INVALID_ARGUMENT)))
    with pytest.raises(RuntimeError, match="capability_mismatch"):
        runtime.chat(
            {
                "input": {
                    "model_id": "gemini-3.5-flash",
                    "messages": [{"role": "user", "content": "hi"}],
                }
            }
        )


@pytest.mark.parametrize(
    ("category", "definite"),
    [
        ("billing_quota_exceeded", True),
        ("model_unavailable", True),
        ("authentication_error", True),
        ("capability_mismatch", False),
    ],
)
def test_new_categories_reach_the_host_as_themselves(category, definite):
    """The host takes errors[0].code when it is one of its categories
    (remote_extension_runtime.safe_remote_error_category)."""
    response = runtime._failure_response(category, request_semantics="mutation")
    assert response["error_code"] == category
    assert response["errors"][0]["code"] == category
    assert response["definitely_no_external_effect"] is definite


def test_main_returns_the_category_of_a_failed_discovery(monkeypatch):
    def failing(request):
        raise runtime.VertexHttpError("billing_quota_exceeded", 403)

    monkeypatch.setitem(main.QUERIES, "llm.list_models", failing)
    monkeypatch.setattr(main, "runtime_context_from_payload", lambda payload: None)
    response, code = main.handle({"mode": "query", "query": {"query_id": "llm.list_models"}})
    assert code == 2
    assert response["errors"][0]["code"] == "billing_quota_exceeded"


# -- B2: only chat models -----------------------------------------------------------------


@pytest.mark.parametrize(
    "model",
    [
        "gemini-3.5-flash",
        "gemini-3.1-pro-preview",
        "gemini-3.5-flash-lite",
        "gemini-4-argon",
        "gemini-3.8-flash",
    ],
)
def test_chat_models_are_kept(model):
    assert runtime.is_chat_model(model)


@pytest.mark.parametrize(
    "model",
    [
        "gemini-2.5-flash-image",
        "gemini-2.5-pro-tts",
        "gemini-live-2.5-flash-native-audio",
        "gemini-embedding-001",
        "gemini-2.5-computer-use-preview-10-2025",
        "gemini-robotics-er-2-preview",
        "gemini-robotics-er-2-preview-info",
        "gemini-3.5-transcribe-live-preview",
        "gemini-3.5-live-translate-preview",
        "gemini-omni-flash-preview",
        "gemini-nano-banana-2.1",
        "gemma-3",
    ],
)
def test_non_chat_models_are_left_out(model):
    assert not runtime.is_chat_model(model)


# -- B3: countTokens decides, and only the project's own failures stop Discover --------------


def _discover(monkeypatch, catalog: dict[str, Any], probe: dict[str, Any], **query):
    """Run list_models with the catalog answer and a countTokens outcome per model."""
    probed = []

    def fake_request(url, credentials, *, body=None, timeout=None, bad_request=None):
        if url.startswith("https://us-central1-aiplatform.googleapis.com/v1beta1/publishers"):
            return catalog
        assert url.endswith(":countTokens") and body == runtime._PROBE_BODY
        assert timeout == runtime.PROBE_TIMEOUT_SECONDS
        model = url.rsplit("/", 1)[1].split(":")[0]
        probed.append(model)
        outcome = probe.get(model, 200)
        if outcome == 200:
            return {"totalTokens": 1}
        if outcome == "timeout":
            raise RuntimeError("timeout")
        if outcome == "slow":
            time.sleep(0.5)
            return {"totalTokens": 1}
        raise runtime.VertexHttpError(runtime.classify_error(outcome, b""), outcome)

    monkeypatch.setattr(
        runtime, "connection", lambda request: ("sample-project", "global", object())
    )
    monkeypatch.setattr(runtime, "request_json", fake_request)
    return runtime.list_models({"query": query}), probed


def catalog(*names: str, token: str = "") -> dict[str, Any]:
    answer: dict[str, Any] = {
        "publisherModels": [{"name": f"publishers/google/models/{n}"} for n in names]
    }
    if token:
        answer["nextPageToken"] = token
    return answer


def test_countTokens_keeps_200_and_drops_404_and_400(monkeypatch):
    result, probed = _discover(
        monkeypatch,
        catalog("gemini-3.5-flash", "gemini-2.5-flash", "gemini-x-flash", "gemini-2.5-pro-tts"),
        {"gemini-2.5-flash": 404, "gemini-x-flash": 400},
    )
    assert [m["model_id"] for m in result["models"]] == ["gemini-3.5-flash"]
    # The speech model is left out by name and never probed.
    assert sorted(probed) == ["gemini-2.5-flash", "gemini-3.5-flash", "gemini-x-flash"]


@pytest.mark.parametrize("outcome", [429, 500, 503, "timeout"])
def test_a_failed_check_keeps_the_model(monkeypatch, outcome):
    result, _ = _discover(
        monkeypatch, catalog("gemini-3.5-flash", "gemini-3.7-flash"), {"gemini-3.7-flash": outcome}
    )
    assert [m["model_id"] for m in result["models"]] == ["gemini-3.5-flash", "gemini-3.7-flash"]


@pytest.mark.parametrize(
    ("status", "category"), [(403, "authentication_error"), (401, "authentication_error")]
)
def test_a_project_failure_stops_discovery_with_its_category(monkeypatch, status, category):
    with pytest.raises(runtime.VertexHttpError, match=category):
        _discover(monkeypatch, catalog("gemini-3.5-flash"), {"gemini-3.5-flash": status})


def test_a_billing_refusal_on_the_check_stops_discovery_as_billing(monkeypatch):
    def fake_request(url, credentials, **kwargs):
        if "v1beta1/publishers" in url:
            return catalog("gemini-3.5-flash")
        raise runtime.VertexHttpError(runtime.classify_error(403, body(BILLING_DISABLED)), 403)

    monkeypatch.setattr(runtime, "connection", lambda request: ("p-roject1", "global", object()))
    monkeypatch.setattr(runtime, "request_json", fake_request)
    with pytest.raises(runtime.VertexHttpError, match="billing_quota_exceeded"):
        runtime.list_models({"query": {}})


def test_checks_run_in_parallel_within_the_budget(monkeypatch):
    names = [f"gemini-{i}-flash" for i in range(runtime.PROBE_WORKERS)]
    started = time.monotonic()
    result, _ = _discover(monkeypatch, catalog(*names), dict.fromkeys(names, "slow"))
    assert len(result["models"]) == len(names)
    assert time.monotonic() - started < 0.5 * len(names) / 2


def test_models_not_checked_within_the_budget_are_kept(monkeypatch):
    monkeypatch.setattr(runtime, "PROBE_BUDGET_SECONDS", 0.05)
    result, _ = _discover(monkeypatch, catalog("gemini-3.5-flash"), {"gemini-3.5-flash": "slow"})
    assert [m["model_id"] for m in result["models"]] == ["gemini-3.5-flash"]


def test_the_budget_fits_the_hosts_discovery_time():
    """Flow Steward gives a discovery 60 s; a page is one catalog read plus the checks."""
    assert runtime.PROBE_TIMEOUT_SECONDS <= 15
    assert runtime.PROBE_BUDGET_SECONDS == 20
    # The host waits for the process to exit, and checks still running when the budget
    # ends finish first: catalog read + budget + one more check must fit in 60 s.
    worst = (
        runtime.PROBE_TIMEOUT_SECONDS + runtime.PROBE_BUDGET_SECONDS + runtime.PROBE_TIMEOUT_SECONDS
    )
    assert worst == 50 < 60


def test_discovery_never_calls_generateContent(monkeypatch):
    _, probed = _discover(monkeypatch, catalog("gemini-3.5-flash"), {})
    assert probed == ["gemini-3.5-flash"]


# -- pagination ---------------------------------------------------------------------------


def test_pagination_cursor_is_kept_across_filtered_pages(monkeypatch):
    first, _ = _discover(monkeypatch, catalog("gemini-2.5-pro-tts", token="t2"), {})
    assert first["models"] == [] and first["next_cursor"] == "0:t2"
    second, _ = _discover(monkeypatch, catalog("gemini-3.5-flash"), {}, cursor="0:t2")
    assert [m["model_id"] for m in second["models"]] == ["gemini-3.5-flash"]
    assert second["next_cursor"] == ""


def test_a_last_page_without_models_after_earlier_ones_is_fine(monkeypatch):
    result, _ = _discover(
        monkeypatch, catalog("gemini-3.5-flash"), {"gemini-3.5-flash": 404}, cursor="1:t9"
    )
    assert result["models"] == [] and result["snapshot_complete"] is True


def test_no_usable_model_in_the_whole_catalog_is_model_unavailable(monkeypatch):
    with pytest.raises(RuntimeError, match="model_unavailable"):
        _discover(monkeypatch, catalog("gemini-2.5-flash"), {"gemini-2.5-flash": 404})


# -- no secrets in errors -----------------------------------------------------------------


def test_errors_carry_no_google_message_or_credential(monkeypatch):
    secret = "credential-material-7f3a9c"
    leaked = {"error": {"code": 403, "message": f"bad key {secret}", "status": "PERMISSION_DENIED"}}
    monkeypatch.setattr(requests, "Session", _session_answering(403, body(leaked)))
    with pytest.raises(runtime.VertexHttpError) as exc:
        runtime.request_json("https://aiplatform.googleapis.com/v1/x", _Credentials())
    response = runtime._failure_response(str(exc.value), request_semantics="safe_read")
    assert secret not in json.dumps(response)
    assert "bad key" not in json.dumps(response)
