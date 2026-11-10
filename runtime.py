"""Vertex Gemini provider operations using account-bound service credentials."""

from __future__ import annotations

import json
import re
from typing import Any

_PROJECT = re.compile(r"[a-z][a-z0-9-]{4,28}[a-z0-9]|[0-9]{6,20}\Z")
_LOCATION = re.compile(r"[a-z][a-z0-9-]{0,62}\Z")
_GOOGLE_OAUTH_ENDPOINT = "https://oauth2.googleapis.com/token"


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
        info = json.loads(raw)
        if not isinstance(info, dict) or info.get("type") != "service_account":
            raise ValueError("service account required")
        if (
            info.get("token_uri") != _GOOGLE_OAUTH_ENDPOINT
            or info.get("universe_domain", "googleapis.com") != "googleapis.com"
        ):
            raise ValueError("untrusted credential endpoint")
        from google.oauth2 import service_account

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
        encoded = json.dumps(continuation, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
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


def request_json(
    url: str, credentials: Any, *, body: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Make one bounded request; never retry an uncertain paid operation."""
    import requests
    from google.auth.exceptions import RefreshError, TransportError
    from google.auth.transport.requests import Request

    session = requests.Session()
    session.trust_env = False
    try:
        auth_request = Request(session=session)

        def bounded_auth_request(*args: Any, **kwargs: Any) -> Any:
            timeout = kwargs.get("timeout")
            kwargs["timeout"] = min(timeout, 20) if timeout is not None else 20
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
            timeout=60,
            allow_redirects=False,
            stream=True,
        )
        if response.status_code in (401, 403):
            raise RuntimeError("authentication_error")
        if response.status_code == 429:
            raise RuntimeError("rate_limited")
        if response.status_code >= 500:
            raise RuntimeError("upstream_unavailable")
        if response.status_code != 200:
            raise RuntimeError("invalid_response")
        raw = bytearray()
        for chunk in response.iter_content(65536):
            raw.extend(chunk)
            if len(raw) > 5 * 1024 * 1024:
                raise RuntimeError("invalid_response")
        data = json.loads(raw)
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


def list_models(request: dict[str, Any]) -> dict[str, Any]:
    from urllib.parse import urlencode

    _, _, credentials = connection(request)
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
        limit = min(max(int(query.get("limit") or 100), 1), 100)
    except (ValueError, TypeError) as exc:
        raise RuntimeError("invalid_response") from exc
    params: dict[str, str | int] = {"pageSize": limit}
    if upstream_cursor:
        params["pageToken"] = upstream_cursor
    url = (
        "https://us-central1-aiplatform.googleapis.com/v1beta1/publishers/google/models?"
        + urlencode(params)
    )
    response = request_json(url, credentials)
    items = response.get("publisherModels")
    if not isinstance(items, list) or not items:
        raise RuntimeError("invalid_response")
    models = []
    for item in items:
        if not isinstance(item, dict):
            raise RuntimeError("invalid_response")
        name = item.get("name", "")
        prefix = "publishers/google/models/"
        if not isinstance(name, str) or not name.startswith(prefix) or len(name) == len(prefix):
            raise RuntimeError("invalid_response")
        model = name[len(prefix) :]
        if not re.fullmatch(r"gemini-[a-z0-9.-]+", model):
            continue
        models.append(
            {
                "model_id": model,
                "label": model,
                "upstream_owner_key": "google",
                "upstream_owner_label": "Google",
                "capabilities": ["chat", "tools", "structured_output"],
            }
        )
    next_token = response.get("nextPageToken") or ""
    if not isinstance(next_token, str) or len(next_token) > 4094:
        raise RuntimeError("invalid_response")
    seen = previously_seen or bool(models)
    if not next_token and not seen:
        raise RuntimeError("invalid_response")
    return {
        "models": models,
        "next_cursor": f"{int(seen)}:{next_token}" if next_token else "",
        "snapshot_complete": True,
    }


def chat(request: dict[str, Any], *, structured: bool = False) -> dict[str, Any]:
    payload = request.get("input") or {}
    project, location, credentials = connection(request)
    model = payload.get("model_id")
    if not isinstance(model, str) or not re.fullmatch(r"gemini-[a-z0-9.-]+", model):
        raise RuntimeError("invalid_configuration")
    body = build_generate_request(payload, structured=structured)
    url = (
        f"{_endpoint(location)}/v1/projects/{project}/locations/{location}"
        f"/publishers/google/models/{model}:generateContent"
    )
    response = request_json(url, credentials, body=body)
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

            output = json.loads(message["content"])
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
        "rate_limited",
        "timeout",
        "upstream_unavailable",
    }
    safe_code = code if code in allowed else "upstream_error"
    definite = request_semantics == "safe_read" or safe_code in {
        "authentication_error",
        "invalid_configuration",
        "invalid_host_envelope",
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
