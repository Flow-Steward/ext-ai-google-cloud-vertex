"""Account-scoped host envelope dispatch."""

import main


def test_query_receives_only_its_account_resource_snapshots(monkeypatch):
    seen = []
    monkeypatch.setitem(
        main.QUERIES, "llm.list_models", lambda request: seen.append(request) or {"models": []}
    )
    payload = {
        "contract_version": "extension_host_v2",
        "mode": "query",
        "query": {"query_id": "llm.list_models", "params": {}},
        "runtime_context": {
            "account_id": "account-1",
            "group_id": None,
            "for_each_id": None,
            "step_id": None,
            "step_execution_id": None,
            "resources": {
                "provider_secrets": {"service_account_json": "opaque-test-secret"},
                "connection_config": {"project_id": "sample-project", "location": "global"},
            },
        },
    }
    response, code = main.handle(payload)
    assert code == 0 and response["ok"]
    assert seen[0]["connection"]["connection_config"] == {
        "project_id": "sample-project",
        "location": "global",
    }
    assert seen[0]["connection"]["credentials"] == {"service_account_json": "opaque-test-secret"}


def test_invalid_envelope_never_invokes_provider(monkeypatch):
    invoked = []
    monkeypatch.setitem(main.QUERIES, "llm.list_models", lambda request: invoked.append(request))
    response, code = main.handle({"mode": "query", "query": {"query_id": "llm.list_models"}})
    assert code == 2 and not response["ok"]
    assert invoked == []
