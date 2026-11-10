"""Real service-account refresh stays within the invocation's Session lifetime."""

from __future__ import annotations

import json
import threading

import requests
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

import runtime

OAUTH_URL = "https://oauth2.googleapis.com/token"
VERTEX_URL = "https://us-central1-aiplatform.googleapis.com/v1/test"


def test_real_service_account_refresh_has_no_background_session_work(monkeypatch):
    monkeypatch.delenv("GOOGLE_AUTH_TRUST_BOUNDARY_ENABLED", raising=False)
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
    _, _, credentials = runtime.connection(
        {
            "connection": {
                "connection_config": {"project_id": "sample-project", "location": "us-central1"},
                "credentials": {"service_account_json": json.dumps(account)},
            }
        }
    )
    calls = []
    closed = threading.Event()

    class Response:
        status_code = 200
        headers = {}

        def __init__(self, data):
            self.content = json.dumps(data).encode()

        def iter_content(self, _size):
            yield self.content

    class Session:
        trust_env = True

        def request(self, method, url, **kwargs):
            calls.append((method, url, kwargs["timeout"]))
            if url == OAUTH_URL:
                return Response(
                    {"access_token": "offline-token", "expires_in": 3600, "token_type": "Bearer"}
                )
            if url == VERTEX_URL:
                assert not self.trust_env
                assert kwargs["headers"]["authorization"] == "Bearer offline-token"
                assert kwargs["allow_redirects"] is False
                return Response({"ok": True})
            # A new SDK background callback must not outlive this invocation.
            assert closed.wait(5), "unexpected authentication request blocked Session closure"
            return Response({"encodedLocations": "0x0", "locations": []})

        def close(self):
            closed.set()

    monkeypatch.setattr(requests, "Session", Session)
    assert runtime.request_json(VERTEX_URL, credentials) == {"ok": True}
    # Join any SDK-owned worker so a delayed callback cannot evade the assertion.
    worker = credentials._rab_manager.refresh_manager._worker
    if worker is not None:
        worker.join(timeout=6)
        assert not worker.is_alive()
    assert closed.is_set()
    assert calls == [("POST", OAUTH_URL, 20), ("GET", VERTEX_URL, 60)]
