"""Vertex Gemini provider operations using account-bound service credentials."""

from __future__ import annotations

import json
import math
import os
import re
from typing import Any

_PROJECT = re.compile(r"[a-z][a-z0-9-]{4,28}[a-z0-9]|[0-9]{6,20}\Z")
_LOCATION = re.compile(r"[a-z][a-z0-9-]{0,62}\Z")
_GOOGLE_OAUTH_ENDPOINT = "https://oauth2.googleapis.com/token"


def _reject_json_constant(_value: str) -> None:
    raise ValueError("invalid JSON number")


def _finite_json_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("invalid JSON number")
    return number


def loads_finite_json(value: str | bytes | bytearray) -> Any:
    """Decode standard JSON without accepting floats outside Python's finite range."""
    return json.loads(value, parse_constant=_reject_json_constant, parse_float=_finite_json_float)


def connection(request: dict[str, Any]) -> tuple[str, str, Any]:
    """Build refreshable credentials solely from the account's declared secret."""
    connection_data = request.get("connection") or {}
    config = connection_data.get("connection_config") or {}
    project, location = config.get("project_id"), config.get("location")
    if not isinstance(project, str) or not _PROJECT.fullmatch(project):
        raise RuntimeError("invalid_configuration")
    if not isinstance(location, str) or not _LOCATION.fullmatch(location):
        raise RuntimeError("invalid_configuration")
    raw = (connection_data.get("credentials") or {}).get("service_account_json")
    if not isinstance(raw, str) or not raw.strip():
        raise RuntimeError("authentication_error")
    try:
        info = loads_finite_json(raw)
        if not isinstance(info, dict) or info.get("type") != "service_account":
            raise ValueError("service account required")
        if (
            info.get("token_uri") != _GOOGLE_OAUTH_ENDPOINT
            or info.get("universe_domain", "googleapis.com") != "googleapis.com"
        ):
            raise ValueError("untrusted credential endpoint")
        from google.auth import environment_vars
        from google.oauth2 import service_account

        # google-auth 2.63 makes RAB opt-in. This extension runs in its own
        # process and keeps lookup enabled by default for service accounts.
        os.environ.setdefault(
            environment_vars.GOOGLE_SDK_EXPERIMENTAL_PYTHON_REGIONAL_ACCESS_BOUNDARY_LOOKUP_ENABLED,
            "true",
        )
        credentials = service_account.Credentials.from_service_account_info(
            info, scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
        # Invocation-owned Sessions cannot be shared with SDK background workers.
        # google-auth preserves this blocking-lookup contract for gcloud: keep RAB
        # discovery enabled, but finish its one fail-fast attempt before disposal.
        credentials._set_blocking_regional_access_boundary_lookup()
    except Exception as exc:
        raise RuntimeError("authentication_error") from exc
    return project, location, credentials


def canonical_message(candidate: dict[str, Any]) -> dict[str, Any]:
    """Preserve opaque Gemini parts for exact tool-call replay."""
    import uuid

    content = candidate.get("content") or {}
    parts = content.get("parts") or []
    if not isinstance(parts, list):
        raise RuntimeError("invalid_response")
    calls = []
    text_parts = []
    replay_parts = []
    for raw in parts:
        if not isinstance(raw, dict):
            raise RuntimeError("invalid_response")
        part = dict(raw)
        call = part.get("functionCall")
        if isinstance(call, dict):
            name, args = call.get("name"), call.get("args")
            if not isinstance(name, str) or not name or not isinstance(args, dict):
                raise RuntimeError("invalid_response")
            call_id = str(call.get("id") or uuid.uuid4())
            calls.append({"id": call_id, "name": name, "arguments": args})
        elif isinstance(part.get("text"), str) and not part.get("thought"):
            text_parts.append(part["text"])
        replay_parts.append(part)
    message: dict[str, Any] = {"role": "assistant", "content": "".join(text_parts)}
    if calls:
        continuation = {"vertex_parts": replay_parts}
        encoded = json.dumps(
            continuation, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        if len(encoded) > 16 * 1024:
            raise RuntimeError("capability_mismatch")
        message["tool_calls"] = calls
        message["continuation"] = continuation
    return message


def vertex_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Replay opaque model parts and order all tool results like the calls."""
    result: list[dict[str, Any]] = []
    pending: list[tuple[str, str, str]] = []
    responses: dict[str, dict[str, Any]] = {}
    for message in messages:
        role = message.get("role")
        if pending and role != "tool":
            raise RuntimeError("capability_mismatch")
        if role == "assistant" and message.get("tool_calls"):
            calls = message["tool_calls"]
            continuation = message.get("continuation") or {}
            parts = continuation.get("vertex_parts") if isinstance(continuation, dict) else None
            if not isinstance(parts, list) or not parts or not isinstance(calls, list):
                raise RuntimeError("capability_mismatch")
            upstream_calls: list[dict[str, Any]] = []
            for part in parts:
                if isinstance(part, dict) and isinstance(part.get("functionCall"), dict):
                    upstream_calls.append(part["functionCall"])
            if len(upstream_calls) != len(calls):
                raise RuntimeError("capability_mismatch")
            for call, upstream in zip(calls, upstream_calls, strict=True):
                if not isinstance(call, dict) or not call.get("id") or not call.get("name"):
                    raise RuntimeError("invalid_response")
                if call["name"] != upstream.get("name") or call.get("arguments") != upstream.get(
                    "args"
                ):
                    raise RuntimeError("capability_mismatch")
                upstream_id = str(upstream.get("id") or "")
                if upstream_id and upstream_id != str(call["id"]):
                    raise RuntimeError("capability_mismatch")
                pending.append((str(call["id"]), str(call["name"]), upstream_id))
            if len({call_id for call_id, _, _ in pending}) != len(pending):
                raise RuntimeError("invalid_response")
            result.append({"role": "model", "parts": parts})
        elif role == "tool":
            call_id = str(message.get("tool_call_id") or "")
            matching = next((item for item in pending if item[0] == call_id), None)
            if matching is None or call_id in responses:
                raise RuntimeError("invalid_response")
            _, name, upstream_id = matching
            function_response: dict[str, Any] = {
                "name": name,
                "response": {"result": message.get("content", "")},
            }
            if upstream_id:
                function_response["id"] = upstream_id
            responses[call_id] = {"functionResponse": function_response}
            if len(responses) == len(pending):
                result.append({"role": "user", "parts": [responses[item[0]] for item in pending]})
                pending.clear()
                responses.clear()
        elif role in {"user", "assistant"}:
            content = message.get("content", "")
            if not isinstance(content, str):
                raise RuntimeError("capability_mismatch")
            result.append(
                {"role": "model" if role == "assistant" else "user", "parts": [{"text": content}]}
            )
        else:
            raise RuntimeError("capability_mismatch")
    if pending:
        raise RuntimeError("capability_mismatch")
    return result


_SCHEMA_KEYS = frozenset(
    {
        "$id",
        "$defs",
        "$ref",
        "$anchor",
        "type",
        "format",
        "title",
        "description",
        "enum",
        "items",
        "prefixItems",
        "minItems",
        "maxItems",
        "minimum",
        "maximum",
        "anyOf",
        "properties",
        "additionalProperties",
        "required",
    }
)


def _check_schema(schema: Any) -> None:
    if isinstance(schema, bool):
        return
    if not isinstance(schema, dict) or set(schema) - _SCHEMA_KEYS:
        raise RuntimeError("capability_mismatch")
    if "$id" in schema or "$anchor" in schema:
        raise RuntimeError("capability_mismatch")
    if "format" in schema:
        from jsonschema import Draft202012Validator

        if schema["format"] not in Draft202012Validator.FORMAT_CHECKER.checkers:
            raise RuntimeError("capability_mismatch")
    ref = schema.get("$ref")
    if ref is not None and (not isinstance(ref, str) or not ref.startswith("#/")):
        raise RuntimeError("capability_mismatch")
    for key in ("properties", "$defs"):
        value = schema.get(key, {})
        if not isinstance(value, dict):
            raise RuntimeError("capability_mismatch")
        for child in value.values():
            _check_schema(child)
    for key in ("items", "additionalProperties"):
        if key in schema:
            _check_schema(schema[key])
    for key in ("anyOf", "prefixItems"):
        if key in schema:
            if not isinstance(schema[key], list):
                raise RuntimeError("capability_mismatch")
            for child in schema[key]:
                _check_schema(child)


def _portable_schema(schema: Any) -> Any:
    if isinstance(schema, bool):
        return schema
    if not isinstance(schema, dict):
        raise RuntimeError("capability_mismatch")
    result = dict(schema)
    if "const" in result:
        constant = result.pop("const")
        if isinstance(constant, bool) or not isinstance(constant, (str, int, float)):
            raise RuntimeError("capability_mismatch")
        if "enum" in result and constant not in result["enum"]:
            raise RuntimeError("capability_mismatch")
        result["enum"] = [constant]
    for key in ("properties", "$defs"):
        if key in result:
            if not isinstance(result[key], dict):
                raise RuntimeError("capability_mismatch")
            result[key] = {name: _portable_schema(child) for name, child in result[key].items()}
    for key in ("items", "additionalProperties"):
        if key in result:
            result[key] = _portable_schema(result[key])
    for key in ("anyOf", "prefixItems"):
        if key in result:
            if not isinstance(result[key], list):
                raise RuntimeError("capability_mismatch")
            result[key] = [_portable_schema(child) for child in result[key]]
    return result


def build_generate_request(payload: dict[str, Any], *, structured: bool = False) -> dict[str, Any]:
    messages = payload.get("messages")
    if not isinstance(messages, list):
        raise RuntimeError("capability_mismatch")
    system = [
        item.get("content")
        for item in messages
        if isinstance(item, dict) and item.get("role") == "system"
    ]
    if any(not isinstance(item, str) for item in system):
        raise RuntimeError("capability_mismatch")
    body: dict[str, Any] = {
        "contents": vertex_messages(
            [item for item in messages if isinstance(item, dict) and item.get("role") != "system"]
        )
    }
    if system:
        body["systemInstruction"] = {"parts": [{"text": "\n\n".join(str(item) for item in system)}]}
    config = {}
    for source, target in (("temperature", "temperature"), ("max_tokens", "maxOutputTokens")):
        if source in payload:
            config[target] = payload[source]
    if structured:
        schema = payload.get("schema")
        if not isinstance(schema, dict) or not schema:
            raise RuntimeError("capability_mismatch")
        upstream_schema = _portable_schema(schema)
        _check_schema(upstream_schema)
        config.update(responseMimeType="application/json", responseJsonSchema=upstream_schema)
    if config:
        body["generationConfig"] = config
    tools = payload.get("tools") or []
    if tools:
        declarations = []
        for tool in tools:
            function = tool.get("function") if isinstance(tool, dict) else None
            if not isinstance(function, dict) or not function.get("name"):
                raise RuntimeError("capability_mismatch")
            declaration = {"name": function["name"], "description": function.get("description", "")}
            if "parameters" in function:
                parameters = _portable_schema(function["parameters"])
                _check_schema(parameters)
                declaration["parametersJsonSchema"] = parameters
            declarations.append(declaration)
        body["tools"] = [{"functionDeclarations": declarations}]
        choice = payload.get("tool_choice", "auto")
        mode = (
            {"auto": "AUTO", "required": "ANY", "none": "NONE"}.get(choice)
            if isinstance(choice, str)
            else None
        )
        if mode is None:
            raise RuntimeError("capability_mismatch")
        body["toolConfig"] = {"functionCallingConfig": {"mode": mode}}
    return body


#: How much of an error answer is read to classify it; the rest is not needed.
MAX_ERROR_BODY_BYTES = 64 * 1024
#: Largest successful answer read.
MAX_ANSWER_BYTES = 5 * 1024 * 1024
#: Seconds one request to Google may take, unless the caller sets less.
REQUEST_TIMEOUT_SECONDS = 60
#: Google's message on a 403 for models a Free trial billing account may not use.
_PAID_BILLING_MESSAGE = "eligible paid billing account"


class VertexHttpError(RuntimeError):
    """A non-200 answer: ``str()`` is the safe category, ``status`` the HTTP status."""

    def __init__(self, category: str, status: int) -> None:
        super().__init__(category)
        self.status = status


def _google_error(raw: bytes) -> tuple[set[str], str]:
    """``error.details[].reason`` values and the lower-cased ``error.message``."""
    try:
        error = loads_finite_json(raw).get("error")
    except (ValueError, AttributeError):
        return set(), ""
    if not isinstance(error, dict):
        return set(), ""
    details = error.get("details")
    reasons = {
        str(detail.get("reason") or "")
        for detail in (details if isinstance(details, list) else [])
        if isinstance(detail, dict)
    }
    return reasons, str(error.get("message") or "").lower()


def _forbidden_category(raw: bytes) -> str:
    """A 403 is about billing only when Google says so; any other 403 is access."""
    reasons, message = _google_error(raw)
    if "BILLING_DISABLED" in reasons or _PAID_BILLING_MESSAGE in message:
        return "billing_quota_exceeded"
    return "authentication_error"


def classify_error(status: int, raw: bytes, *, bad_request: str = "invalid_response") -> str:
    """The Flow Steward category for one Google error answer.

    Only ``error.details[].reason`` and ``error.message`` of a 403 are read, to tell
    billing from access; none of Google's text leaves this module. ``bad_request``
    is the category of a 400 for this kind of request.
    """
    match status:
        case 400:
            return bad_request
        case 401:
            return "authentication_error"
        case 403:
            return _forbidden_category(raw)
        case 404:
            return "model_unavailable"
        case 429:
            return "rate_limited"
        case _ if status >= 500:
            return "upstream_unavailable"
        case _:
            return "invalid_response"


def _read_bounded(response: Any, limit: int) -> bytes:
    raw = bytearray()
    for chunk in response.iter_content(65536):
        raw.extend(chunk)
        if len(raw) > limit:
            break
    return bytes(raw[:limit])


def request_json(
    url: str,
    credentials: Any,
    *,
    body: dict[str, Any] | None = None,
    timeout: float = REQUEST_TIMEOUT_SECONDS,
    bad_request: str = "invalid_response",
) -> dict[str, Any]:
    """Make one bounded request; never retry an uncertain paid operation."""
    import requests
    from google.auth.exceptions import RefreshError, TransportError
    from google.auth.transport.requests import Request

    session = requests.Session()
    session.trust_env = False
    try:
        auth_request = Request(session=session)

        timeout_seconds = timeout

        def bounded_auth_request(*args: Any, **kwargs: Any) -> Any:
            timeout = kwargs.get("timeout")
            limit = min(20, timeout_seconds)
            kwargs["timeout"] = min(timeout, limit) if timeout is not None else limit
            return auth_request(*args, **kwargs)

        headers = {"Content-Type": "application/json"}
        credentials.before_request(
            bounded_auth_request, "POST" if body is not None else "GET", url, headers
        )
        response = session.request(
            "POST" if body is not None else "GET",
            url,
            headers=headers,
            json=body,
            timeout=timeout,
            allow_redirects=False,
            stream=True,
        )
        if response.status_code != 200:
            error_body = _read_bounded(response, MAX_ERROR_BODY_BYTES)
            raise VertexHttpError(
                classify_error(response.status_code, error_body, bad_request=bad_request),
                response.status_code,
            )
        raw = bytearray()
        for chunk in response.iter_content(65536):
            raw.extend(chunk)
            if len(raw) > MAX_ANSWER_BYTES:
                raise RuntimeError("invalid_response")
        data = loads_finite_json(raw)
        if not isinstance(data, dict):
            raise RuntimeError("invalid_response")
        return data
    except RefreshError as exc:
        raise RuntimeError("authentication_error") from exc
    except TransportError as exc:
        category = (
            "timeout" if isinstance(exc.__cause__, requests.Timeout) else "upstream_unavailable"
        )
        raise RuntimeError(category) from exc
    except requests.Timeout as exc:
        raise RuntimeError("timeout") from exc
    except requests.RequestException as exc:
        raise RuntimeError("upstream_unavailable") from exc
    except (ValueError, TypeError) as exc:
        raise RuntimeError("invalid_response") from exc
    finally:
        session.close()


def _endpoint(location: str) -> str:
    if location == "global":
        return "https://aiplatform.googleapis.com"
    if location in {"us", "eu"}:
        return f"https://aiplatform.{location}.rep.googleapis.com"
    return f"https://{location}-aiplatform.googleapis.com"


#: Name segments of Gemini models that are not chat models (speech, live audio, image,
#: embeddings, computer use, robotics, transcription, translation). The catalog
#: (PublisherModel) has no field for modalities or supported methods, so the name is
#: the only signal; countTokens below cannot tell them apart either (they answer 200).
NON_CHAT_SEGMENTS = frozenset(
    {
        "audio",
        "banana",
        "computer",
        "embedding",
        "image",
        "live",
        "omni",
        "robotics",
        "transcribe",
        "translate",
        "tts",
    }
)
#: Suffix of catalog entries that describe a model rather than serve one.
INFO_SUFFIX = "-info"
_GEMINI_MODEL = re.compile(r"gemini-[a-z0-9.-]+")
#: Catalog page size Google accepts at most.
MAX_PAGE_SIZE = 100
#: countTokens requests in flight at once while Discover checks models.
PROBE_WORKERS = 8
#: Seconds one countTokens or catalog request may take.
PROBE_TIMEOUT_SECONDS = 15
#: Seconds all countTokens checks of one page may take; models not checked in time are
#: kept, since a slow answer says nothing about the model. Flow Steward gives the whole
#: discovery 60 s (llm_provider_extension_service._DISCOVERY_TIME_BUDGET_SECONDS) and
#: waits for this process to exit (extension_runtime_invocation._wait_for_exit). Checks
#: already running when the budget ends are not interrupted: the interpreter joins their
#: threads at exit. Worst case per page: catalog read 15 s + this budget 20 s + one
#: check still running 15 s = 50 s, inside 60 s.
PROBE_BUDGET_SECONDS = 20
#: Answers that mean the project cannot reach the model in this location.
_DROPPED_STATUSES = frozenset({400, 404})
#: Answers that are about the project, not the model: Discover stops with them.
_PROJECT_STATUSES = frozenset({401, 403})
_PROBE_BODY = {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]}


def is_chat_model(model_id: str) -> bool:
    """Whether a Gemini model id names a chat model, by its name segments."""
    if not _GEMINI_MODEL.fullmatch(model_id) or model_id.endswith(INFO_SUFFIX):
        return False
    return not NON_CHAT_SEGMENTS.intersection(model_id.split("-"))


def _model_url(project: str, location: str, model: str, method: str) -> str:
    return (
        f"{_endpoint(location)}/v1/projects/{project}/locations/{location}"
        f"/publishers/google/models/{model}:{method}"
    )


def _reachable(project: str, location: str, model: str, credentials: Any) -> bool:
    """countTokens (free) for one model: False when Google says the project cannot use it.

    A failure about the project (401, 403) is raised; a rate limit, server error or
    timeout keeps the model, since it says nothing about the model.
    """
    try:
        request_json(
            _model_url(project, location, model, "countTokens"),
            credentials,
            body=_PROBE_BODY,
            timeout=PROBE_TIMEOUT_SECONDS,
        )
    except VertexHttpError as exc:
        if exc.status in _DROPPED_STATUSES:
            return False
        if exc.status in _PROJECT_STATUSES:
            raise
        return True
    except RuntimeError:
        return True
    return True


def reachable_models(project: str, location: str, models: list[str], credentials: Any) -> list[str]:
    """The models of ``models`` this project can call in ``location``, in order."""
    import concurrent.futures

    if not models:
        return []
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=PROBE_WORKERS)
    try:
        futures = {
            pool.submit(_reachable, project, location, model, credentials): model
            for model in models
        }
        done, _ = concurrent.futures.wait(futures, timeout=PROBE_BUDGET_SECONDS)
        dropped = set()
        for future in done:
            if not future.result():
                dropped.add(futures[future])
        return [model for model in models if model not in dropped]
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def list_models(request: dict[str, Any]) -> dict[str, Any]:
    from urllib.parse import urlencode

    project, location, credentials = connection(request)
    query = request.get("query") or {}
    cursor = str(query.get("cursor") or "")
    previously_seen = False
    upstream_cursor = ""
    if cursor:
        marker, separator, upstream_cursor = cursor.partition(":")
        if marker not in {"0", "1"} or not separator or not upstream_cursor or len(cursor) > 4096:
            raise RuntimeError("invalid_response")
        previously_seen = marker == "1"
    try:
        limit = min(max(int(query.get("limit") or MAX_PAGE_SIZE), 1), MAX_PAGE_SIZE)
    except (ValueError, TypeError) as exc:
        raise RuntimeError("invalid_response") from exc
    params: dict[str, str | int] = {"pageSize": limit}
    if upstream_cursor:
        params["pageToken"] = upstream_cursor
    # The publishers catalog is global; us-central1 serves it for every location.
    url = (
        "https://us-central1-aiplatform.googleapis.com/v1beta1/publishers/google/models?"
        + urlencode(params)
    )
    response = request_json(url, credentials, timeout=PROBE_TIMEOUT_SECONDS)
    items = response.get("publisherModels")
    if not isinstance(items, list) or not items:
        raise RuntimeError("invalid_response")
    prefix = "publishers/google/models/"
    candidates = []
    for item in items:
        name = item.get("name", "") if isinstance(item, dict) else None
        if not isinstance(name, str) or not name.startswith(prefix) or len(name) == len(prefix):
            raise RuntimeError("invalid_response")
        model = name[len(prefix) :]
        if is_chat_model(model):
            candidates.append(model)
    models = [
        {
            "model_id": model,
            "label": model,
            "upstream_owner_key": "google",
            "upstream_owner_label": "Google",
            "capabilities": ["chat", "tools", "structured_output"],
        }
        for model in reachable_models(project, location, candidates, credentials)
    ]
    next_token = response.get("nextPageToken") or ""
    if not isinstance(next_token, str) or len(next_token) > 4094:
        raise RuntimeError("invalid_response")
    seen = previously_seen or bool(models)
    if not next_token and not seen:
        # The whole catalog was read and the project can use none of its chat models
        # in this location.
        raise RuntimeError("model_unavailable")
    return {
        "models": models,
        "next_cursor": f"{int(seen)}:{next_token}" if next_token else "",
        "snapshot_complete": True,
    }


def chat(request: dict[str, Any], *, structured: bool = False) -> dict[str, Any]:
    payload = request.get("input") or {}
    project, location, credentials = connection(request)
    model = payload.get("model_id")
    if not isinstance(model, str) or not _GEMINI_MODEL.fullmatch(model):
        raise RuntimeError("invalid_configuration")
    body = build_generate_request(payload, structured=structured)
    url = _model_url(project, location, model, "generateContent")
    response = request_json(url, credentials, body=body, bad_request="capability_mismatch")
    candidates = response.get("candidates")
    if (
        not isinstance(candidates, list)
        or len(candidates) != 1
        or not isinstance(candidates[0], dict)
    ):
        raise RuntimeError("invalid_response")
    candidate = candidates[0]
    if candidate.get("finishReason") != "STOP":
        raise RuntimeError("invalid_response")
    message = canonical_message(candidate)
    calls = message.get("tool_calls", [])
    if not calls and not str(message.get("content") or "").strip():
        raise RuntimeError("invalid_response")
    if payload.get("parallel_tool_calls") is False and len(calls) > 1:
        raise RuntimeError("capability_mismatch")
    if (payload.get("tool_choice") == "none" and calls) or (
        payload.get("tool_choice") == "required" and not calls
    ):
        raise RuntimeError("capability_mismatch")
    if calls:
        import jsonschema

        declared = {
            item["function"]["name"]: item["function"]["parameters"]
            for item in payload.get("tools", [])
            if isinstance(item, dict)
            and isinstance(item.get("function"), dict)
            and isinstance(item["function"].get("parameters"), dict)
            and isinstance(item["function"].get("name"), str)
        }
        for call in calls:
            schema = declared.get(call["name"])
            if schema is None:
                raise RuntimeError("capability_mismatch")
            try:
                jsonschema.Draft202012Validator.check_schema(schema)
                jsonschema.Draft202012Validator(
                    schema, format_checker=jsonschema.Draft202012Validator.FORMAT_CHECKER
                ).validate(call["arguments"])
            except (
                jsonschema.exceptions.ValidationError,
                jsonschema.exceptions.SchemaError,
            ) as exc:
                raise RuntimeError("capability_mismatch") from exc
    usage = response.get("usageMetadata") or {}
    result: dict[str, Any] = {
        "message": message,
        "finish_reason": candidate.get("finishReason", ""),
        "usage": {
            "input_tokens": usage.get("promptTokenCount", 0),
            "output_tokens": usage.get("candidatesTokenCount", 0),
            "total_tokens": usage.get("totalTokenCount", 0),
        },
    }
    response_id = response.get("responseId")
    if isinstance(response_id, str) and re.fullmatch(r"[A-Za-z0-9._:-]{1,200}", response_id):
        result["upstream_request_id"] = response_id
    if structured:
        try:
            import jsonschema

            output = loads_finite_json(message["content"])
            jsonschema.Draft202012Validator.check_schema(payload["schema"])
            jsonschema.Draft202012Validator(
                payload["schema"], format_checker=jsonschema.Draft202012Validator.FORMAT_CHECKER
            ).validate(output)
        except (
            ValueError,
            jsonschema.exceptions.ValidationError,
            jsonschema.exceptions.SchemaError,
        ) as exc:
            raise RuntimeError("capability_mismatch") from exc
        result["structured_output"] = output
    return result


def _failure_response(code: str, *, request_semantics: str) -> dict[str, Any]:
    allowed = {
        "authentication_error",
        "invalid_configuration",
        "invalid_response",
        "invalid_host_envelope",
        "capability_mismatch",
        "model_unavailable",
        "billing_quota_exceeded",
        "rate_limited",
        "timeout",
        "upstream_unavailable",
    }
    safe_code = code if code in allowed else "upstream_error"
    definite = request_semantics == "safe_read" or safe_code in {
        "authentication_error",
        "invalid_configuration",
        "invalid_host_envelope",
        "model_unavailable",
        "billing_quota_exceeded",
        "rate_limited",
    }
    message = f"Vertex request failed: {safe_code}"
    return {
        "ok": False,
        "error_code": safe_code,
        "error": message,
        "errors": [{"code": safe_code, "message": message}],
        "failure_class": "transient"
        if safe_code in {"timeout", "upstream_unavailable", "upstream_error"}
        else "provider",
        "retryable": definite
        and safe_code in {"rate_limited", "timeout", "upstream_unavailable", "upstream_error"},
        "definitely_no_external_effect": definite,
        "external_effect_status": "failed" if definite else "timeout_unknown",
        "provider_error_code": safe_code,
    }
