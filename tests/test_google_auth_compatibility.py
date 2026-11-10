"""Real service-account refresh stays within the invocation's Session lifetime."""

from __future__ import annotations

import json
import threading

import pytest
import requests
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

import runtime

OAUTH_URL = "https://oauth2.googleapis.com/token"
VERTEX_URL = "https://us-central1-aiplatform.googleapis.com/v1/test"
IAM_URL = (
    "https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/"
    "offline@sample-project.iam.gserviceaccount.com/allowedLocations"
)


@pytest.fixture
def credentials():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    account = {
        "type": "service_account",
        "project_id": "sample-project",
        "private_key_id": "offline-test-key",
        "private_key": key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        ).decode(),
        "client_email": "offline@sample-project.iam.gserviceaccount.com",
        "client_id": "123456789",
        "token_uri": OAUTH_URL,
    }
    return runtime.connection(
        {
            "connection": {
                "connection_config": {"project_id": "sample-project", "location": "us-central1"},
                "credentials": {"service_account_json": json.dumps(account)},
            }
        }
    )[2]


@pytest.mark.parametrize("lookup_result", ["success", "timeout", "unavailable", "forbidden"])
@pytest.mark.parametrize("vertex_status", [200, 500])
def test_real_service_account_lookup_finishes_before_session_close(
    monkeypatch, credentials, lookup_result, vertex_status
):
    calls = []
    closed = threading.Event()
    owner = threading.get_ident()

    class Response:
        headers = {}

        def __init__(self, data, status=200):
            self.content = json.dumps(data).encode()
            self.status_code = status

        def iter_content(self, _size):
            yield self.content

    class Session:
        trust_env = True

        def request(self, method, url, **kwargs):
            assert not closed.is_set(), "authentication used a disposed Session"
            assert threading.get_ident() == owner, "authentication escaped the invocation thread"
            assert not self.trust_env
            calls.append((method, url, kwargs["timeout"]))
            if url == OAUTH_URL:
                return Response(
                    {"access_token": "offline-token", "expires_in": 3600, "token_type": "Bearer"}
                )
            assert kwargs["headers"]["authorization"] == "Bearer offline-token"
            if url == IAM_URL:
                if lookup_result == "timeout":
                    raise requests.Timeout("offline IAM timeout")
                if lookup_result in {"unavailable", "forbidden"}:
                    return Response(
                        {"error": "offline IAM error"},
                        503 if lookup_result == "unavailable" else 403,
                    )
                return Response({"encodedLocations": "0x123", "locations": ["us-central1"]})
            assert url == VERTEX_URL
            assert kwargs["allow_redirects"] is False
            assert kwargs["headers"].get("x-allowed-locations") == (
                "0x123" if lookup_result == "success" else None
            )
            return Response({"ok": True}, vertex_status)

        def close(self):
            closed.set()

    monkeypatch.setattr(requests, "Session", Session)
    if vertex_status == 200:
        assert runtime.request_json(VERTEX_URL, credentials) == {"ok": True}
    else:
        with pytest.raises(RuntimeError, match="upstream_unavailable"):
            runtime.request_json(VERTEX_URL, credentials)
    assert closed.is_set()
    assert calls == [("POST", OAUTH_URL, 20), ("GET", IAM_URL, 3), ("GET", VERTEX_URL, 60)]
    # No SDK-owned background work may remain after success or failure.
    worker = credentials._rab_manager.refresh_manager._worker
    assert worker is None


def test_real_service_account_regional_endpoint_needs_no_lookup(monkeypatch, credentials):
    url = "https://aiplatform.us.rep.googleapis.com/v1/test"
    calls = []
    closed = []

    class Response:
        status_code = 200
        headers = {}
        content = b'{"access_token":"offline-token","expires_in":3600,"token_type":"Bearer"}'

        def iter_content(self, _size):
            yield b'{"ok":true}'

    class Session:
        trust_env = True

        def request(self, method, request_url, **kwargs):
            assert not closed
            calls.append((method, request_url))
            return Response()

        def close(self):
            closed.append(True)

    monkeypatch.setattr(requests, "Session", Session)
    assert runtime.request_json(url, credentials) == {"ok": True}
    assert calls == [("POST", OAUTH_URL), ("GET", url)]
    assert closed
