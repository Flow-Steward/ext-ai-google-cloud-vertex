"""Contract checks for account isolation and Gemini tool continuation."""

import pytest

import runtime


def test_missing_account_credentials_never_use_ambient_identity():
    with pytest.raises(RuntimeError, match="authentication_error"):
        runtime.connection(
            {
                "connection": {
                    "connection_config": {
                        "project_id": "sample-project",
                        "location": "us-central1",
                    },
                    "credentials": {},
                }
            }
        )


def test_function_call_signature_is_replayed_with_result():
    candidate = {
        "content": {
            "role": "model",
            "parts": [
                {
                    "functionCall": {
                        "name": "lookup",
                        "args": {"x": 1},
                        "id": "call-1",
                    },
                    "thoughtSignature": "opaque",
                }
            ],
        }
    }
    message = runtime.canonical_message(candidate)
    request = runtime.vertex_messages(
        [message, {"role": "tool", "tool_call_id": "call-1", "content": "ok"}]
    )
    assert request[0]["parts"][0]["thoughtSignature"] == "opaque"
    assert request[1]["parts"][0]["functionResponse"]["name"] == "lookup"
    assert request[1]["parts"][0]["functionResponse"]["response"] == {"result": "ok"}


def test_rejects_untrusted_token_endpoint_before_refresh():
    credential = '{"type":"service_account","token_uri":"https://evil.example/token"}'
    request = {
        "connection": {
            "connection_config": {
                "project_id": "sample-project",
                "location": "us-central1",
            },
            "credentials": {"service_account_json": credential},
        }
    }
    with pytest.raises(RuntimeError, match="authentication_error"):
        runtime.connection(request)


def test_rejects_path_in_location():
    credential = '{"type":"service_account","token_uri":"https://oauth2.googleapis.com/token"}'
    request = {
        "connection": {
            "connection_config": {
                "project_id": "sample-project",
                "location": "us-central1/../global",
            },
            "credentials": {"service_account_json": credential},
        }
    }
    with pytest.raises(RuntimeError, match="invalid_configuration"):
        runtime.connection(request)


def test_structured_request_uses_json_schema():
    payload = {
        "model_id": "gemini-2.5-flash",
        "messages": [{"role": "user", "content": "Return a value"}],
        "schema": {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
    }
    body = runtime.build_generate_request(payload, structured=True)
    assert body["generationConfig"]["responseMimeType"] == "application/json"
    assert body["generationConfig"]["responseJsonSchema"] == payload["schema"]


def test_tool_declaration_and_choice_are_translated():
    payload = {
        "model_id": "gemini-2.5-flash",
        "messages": [{"role": "user", "content": "Use a tool"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Look up",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        "tool_choice": "required",
    }
    body = runtime.build_generate_request(payload)
    assert body["tools"][0]["functionDeclarations"][0]["name"] == "lookup"
    assert body["toolConfig"]["functionCallingConfig"]["mode"] == "ANY"


def test_chat_translates_usage_and_no_retry(monkeypatch):
    calls = []
    monkeypatch.setattr(
        runtime, "connection", lambda request: ("sample-project", "us-central1", object())
    )

    def fake_request(url, credentials, *, body=None):
        calls.append((url, body))
        return {
            "candidates": [{"content": {"parts": [{"text": "hello"}]}, "finishReason": "STOP"}],
            "usageMetadata": {
                "promptTokenCount": 2,
                "candidatesTokenCount": 3,
                "totalTokenCount": 5,
            },
        }

    monkeypatch.setattr(runtime, "request_json", fake_request)
    result = runtime.chat(
        {"input": {"model_id": "gemini-2.5-flash", "messages": [{"role": "user", "content": "hi"}]}}
    )
    assert result["message"]["content"] == "hello"
    assert result["usage"]["total_tokens"] == 5
    assert len(calls) == 1
    assert (
        "/projects/sample-project/locations/us-central1/publishers/google/models/gemini-2.5-flash:generateContent"
        in calls[0][0]
    )


def test_discovery_maps_only_gemini_publisher_models(monkeypatch):
    monkeypatch.setattr(
        runtime, "connection", lambda request: ("sample-project", "global", object())
    )
    monkeypatch.setattr(
        runtime,
        "request_json",
        lambda *args, **kwargs: {
            "publisherModels": [
                {"name": "publishers/google/models/gemini-2.5-flash"},
                {"name": "publishers/google/models/gemma"},
            ]
        },
    )
    result = runtime.list_models({"query": {}})
    assert [model["model_id"] for model in result["models"]] == ["gemini-2.5-flash"]


def test_timeout_after_chat_is_not_marked_safe_to_retry():
    error = runtime._failure_response("timeout", request_semantics="mutation")
    assert error["retryable"] is False
    assert error["definitely_no_external_effect"] is False


def test_post_response_capability_failure_is_not_marked_no_effect():
    error = runtime._failure_response("capability_mismatch", request_semantics="mutation")
    assert error["definitely_no_external_effect"] is False


def test_replay_without_opaque_model_parts_fails():
    with pytest.raises(RuntimeError, match="capability_mismatch"):
        runtime.vertex_messages(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"id": "x", "name": "lookup", "arguments": {}}],
                }
            ]
        )


def test_const_schema_is_rewritten_without_loosening():
    schema = {
        "type": "object",
        "properties": {"status": {"type": "string", "const": "ok"}},
        "required": ["status"],
    }
    body = runtime.build_generate_request(
        {"messages": [{"role": "user", "content": "Go"}], "schema": schema}, structured=True
    )
    assert body["generationConfig"]["responseJsonSchema"]["properties"]["status"]["enum"] == ["ok"]
    assert schema["properties"]["status"]["const"] == "ok"


def test_structured_output_must_match_original_schema(monkeypatch):
    monkeypatch.setattr(
        runtime, "connection", lambda request: ("sample-project", "global", object())
    )
    monkeypatch.setattr(
        runtime,
        "request_json",
        lambda *args, **kwargs: {
            "candidates": [
                {"content": {"parts": [{"text": '{"status":"bad"}'}]}, "finishReason": "STOP"}
            ]
        },
    )
    request = {
        "input": {
            "model_id": "gemini-2.5-flash",
            "messages": [{"role": "user", "content": "go"}],
            "schema": {
                "type": "object",
                "properties": {"status": {"type": "string", "const": "ok"}},
                "required": ["status"],
            },
        }
    }
    with pytest.raises(RuntimeError, match="capability_mismatch"):
        runtime.chat(request, structured=True)


def test_structured_schema_rejects_remote_reference_before_call():
    with pytest.raises(RuntimeError, match="capability_mismatch"):
        runtime.build_generate_request(
            {
                "messages": [{"role": "user", "content": "go"}],
                "schema": {"$ref": "https://example.invalid/schema"},
            },
            structured=True,
        )


def test_discovery_rejects_missing_or_malformed_publisher_list(monkeypatch):
    monkeypatch.setattr(
        runtime, "connection", lambda request: ("sample-project", "global", object())
    )
    for response in ({}, {"publisherModels": [None]}, {"publisherModels": []}):
        monkeypatch.setattr(
            runtime, "request_json", lambda *args, response=response, **kwargs: response
        )
        with pytest.raises(RuntimeError, match="invalid_response"):
            runtime.list_models({"query": {}})


@pytest.mark.parametrize("reason", ["SAFETY", "MAX_TOKENS", "RECITATION"])
def test_chat_refuses_unsuccessful_finish_reasons(monkeypatch, reason):
    monkeypatch.setattr(
        runtime, "connection", lambda request: ("sample-project", "global", object())
    )
    monkeypatch.setattr(
        runtime,
        "request_json",
        lambda *args, **kwargs: {
            "candidates": [{"content": {"parts": [{"text": "partial"}]}, "finishReason": reason}]
        },
    )
    with pytest.raises(RuntimeError, match="invalid_response"):
        runtime.chat(
            {
                "input": {
                    "model_id": "gemini-2.5-flash",
                    "messages": [{"role": "user", "content": "hi"}],
                }
            }
        )


def test_discovery_does_not_replace_catalog_with_no_gemini_models(monkeypatch):
    monkeypatch.setattr(
        runtime, "connection", lambda request: ("sample-project", "global", object())
    )
    monkeypatch.setattr(
        runtime,
        "request_json",
        lambda *args, **kwargs: {"publisherModels": [{"name": "publishers/google/models/gemma"}]},
    )
    with pytest.raises(RuntimeError, match="invalid_response"):
        runtime.list_models({"query": {}})


def test_parallel_tool_results_are_grouped_and_keep_generated_ids():
    candidate = {
        "content": {
            "parts": [
                {"functionCall": {"name": "first", "args": {"x": 1}}, "thoughtSignature": "signed"},
                {"functionCall": {"name": "second", "args": {"x": 2}}},
            ]
        }
    }
    message = runtime.canonical_message(candidate)
    first, second = message["tool_calls"]
    assert first["id"] and second["id"] and first["id"] != second["id"]
    turns = runtime.vertex_messages(
        [
            message,
            {"role": "tool", "tool_call_id": first["id"], "content": "one"},
            {"role": "tool", "tool_call_id": second["id"], "content": "two"},
        ]
    )
    assert turns[0]["parts"][0]["thoughtSignature"] == "signed"
    assert [part["functionResponse"]["name"] for part in turns[1]["parts"]] == ["first", "second"]
    assert len(turns) == 2


def test_chat_preserves_google_response_id(monkeypatch):
    monkeypatch.setattr(
        runtime, "connection", lambda request: ("sample-project", "global", object())
    )
    monkeypatch.setattr(
        runtime,
        "request_json",
        lambda *args, **kwargs: {
            "responseId": "response-123",
            "candidates": [{"content": {"parts": [{"text": "hello"}]}, "finishReason": "STOP"}],
        },
    )
    result = runtime.chat(
        {"input": {"model_id": "gemini-2.5-flash", "messages": [{"role": "user", "content": "hi"}]}}
    )
    assert result["upstream_request_id"] == "response-123"


def test_refresh_failure_is_classified_without_exposing_credential_text():
    from google.auth.exceptions import RefreshError

    class BrokenCredentials:
        def before_request(self, *args, **kwargs):
            raise RefreshError("secret sentinel must not escape")

    with pytest.raises(RuntimeError, match="authentication_error") as exc:
        runtime.request_json("https://aiplatform.googleapis.com/v1/test", BrokenCredentials())
    assert "secret sentinel" not in str(exc.value)


def test_structured_output_validates_string_format(monkeypatch):
    monkeypatch.setattr(
        runtime, "connection", lambda request: ("sample-project", "global", object())
    )
    monkeypatch.setattr(
        runtime,
        "request_json",
        lambda *args, **kwargs: {
            "candidates": [
                {
                    "content": {"parts": [{"text": '{"email":"not-an-email"}'}]},
                    "finishReason": "STOP",
                }
            ]
        },
    )
    request = {
        "input": {
            "model_id": "gemini-2.5-flash",
            "messages": [{"role": "user", "content": "go"}],
            "schema": {
                "type": "object",
                "properties": {"email": {"type": "string", "format": "email"}},
                "required": ["email"],
            },
        }
    }
    with pytest.raises(RuntimeError, match="capability_mismatch"):
        runtime.chat(request, structured=True)


def test_parallel_results_are_reordered_to_original_call_order():
    message = runtime.canonical_message(
        {
            "content": {
                "parts": [
                    {"functionCall": {"name": "same", "args": {"x": 1}}},
                    {"functionCall": {"name": "same", "args": {"x": 2}}},
                ]
            }
        }
    )
    first, second = message["tool_calls"]
    turns = runtime.vertex_messages(
        [
            message,
            {"role": "tool", "tool_call_id": second["id"], "content": "second"},
            {"role": "tool", "tool_call_id": first["id"], "content": "first"},
        ]
    )
    assert [part["functionResponse"]["response"]["result"] for part in turns[1]["parts"]] == [
        "first",
        "second",
    ]


def test_tool_parameters_preserve_verifier_constant_constraint():
    payload = {
        "messages": [{"role": "user", "content": "Use tool"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "probe",
                    "parameters": {
                        "type": "object",
                        "properties": {"key": {"type": "string", "const": "probe"}},
                        "required": ["key"],
                    },
                },
            }
        ],
        "tool_choice": "required",
    }
    body = runtime.build_generate_request(payload)
    schema = body["tools"][0]["functionDeclarations"][0]["parametersJsonSchema"]
    assert schema["properties"]["key"]["enum"] == ["probe"]


def test_chat_rejects_tool_call_that_violates_declared_schema(monkeypatch):
    monkeypatch.setattr(
        runtime, "connection", lambda request: ("sample-project", "global", object())
    )
    monkeypatch.setattr(
        runtime,
        "request_json",
        lambda *args, **kwargs: {
            "candidates": [
                {
                    "content": {
                        "parts": [{"functionCall": {"name": "probe", "args": {"key": "wrong"}}}]
                    },
                    "finishReason": "STOP",
                }
            ]
        },
    )
    request = {
        "input": {
            "model_id": "gemini-2.5-flash",
            "messages": [{"role": "user", "content": "go"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "probe",
                        "parameters": {
                            "type": "object",
                            "properties": {"key": {"type": "string", "const": "probe"}},
                            "required": ["key"],
                        },
                    },
                }
            ],
        }
    }
    with pytest.raises(RuntimeError, match="capability_mismatch"):
        runtime.chat(request)


def test_oversized_tool_continuation_is_rejected_before_it_can_be_lost():
    candidate = {
        "content": {
            "parts": [
                {"functionCall": {"name": "probe", "args": {}}, "thoughtSignature": "x" * 17000}
            ]
        }
    }
    with pytest.raises(RuntimeError, match="capability_mismatch"):
        runtime.canonical_message(candidate)


def test_request_timeout_is_not_retried(monkeypatch):
    import requests

    calls = []

    class Session:
        trust_env = True

        def request(self, *args, **kwargs):
            calls.append((args, kwargs))
            raise requests.Timeout("private diagnostic")

        def close(self):
            pass

    class Credentials:
        def before_request(self, *args):
            pass

    monkeypatch.setattr(requests, "Session", Session)
    with pytest.raises(RuntimeError, match="timeout") as exc:
        runtime.request_json("https://aiplatform.googleapis.com/v1/test", Credentials(), body={})
    assert len(calls) == 1
    assert "private diagnostic" not in str(exc.value)


def test_chat_rejects_success_without_text_or_tool_calls(monkeypatch):
    monkeypatch.setattr(
        runtime, "connection", lambda request: ("sample-project", "global", object())
    )
    monkeypatch.setattr(
        runtime,
        "request_json",
        lambda *args, **kwargs: {
            "candidates": [{"content": {"parts": []}, "finishReason": "STOP"}]
        },
    )
    with pytest.raises(RuntimeError, match="invalid_response"):
        runtime.chat(
            {
                "input": {
                    "model_id": "gemini-2.5-flash",
                    "messages": [{"role": "user", "content": "hi"}],
                }
            }
        )


@pytest.mark.parametrize(
    ("location", "expected"),
    [
        ("us", "https://aiplatform.us.rep.googleapis.com"),
        ("eu", "https://aiplatform.eu.rep.googleapis.com"),
        ("global", "https://aiplatform.googleapis.com"),
        ("us-central1", "https://us-central1-aiplatform.googleapis.com"),
    ],
)
def test_vertex_endpoint_matches_location_type(location, expected):
    assert runtime._endpoint(location) == expected
