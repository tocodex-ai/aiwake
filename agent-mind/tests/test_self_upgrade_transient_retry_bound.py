"""Regression test for the self-upgrade transient-failure retry bound.

Evidence (online, 2026-06-16): two proposals were stuck in 'approved' forever
(upg_20260605T143259_904c6f78c8 created 6/5, upg_20260614T232446_99546db351
created 6/14). Root cause: in evolution/self_upgrade.py the transient
generation-failure branch of generate_and_apply_patch (empty LLM response,
declined=False) only appended a note and kept status='approved', so the engine
re-selected the proposal endlessly without ever retrying productively or
giving up.

This test asserts the bounded-retry + terminal-downgrade behaviour:
- each transient failure increments patch_attempt_count
- status stays 'approved' while under MAX_PATCH_ATTEMPTS
- once attempts reach MAX_PATCH_ATTEMPTS the proposal is downgraded to 'blocked'
  so the engine stops re-selecting it.

Run from repository root:
    python src/agent-mind/tests/test_self_upgrade_transient_retry_bound.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path


def _find_agent_dir() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "heartbeat.py").exists() and (parent / "tool_router.py").exists():
            return parent
        if parent.name == "agent-mind":
            return parent
        candidate = parent / "src" / "agent-mind"
        if candidate.exists():
            return candidate
        candidate = parent / "agent-mind"
        if candidate.exists():
            return candidate
    raise RuntimeError("Cannot locate agent-mind directory")


AGENT_DIR = _find_agent_dir()
sys.path.insert(0, str(AGENT_DIR))


def _test_internal_empty_response_retry(su) -> None:  # noqa: ANN001
    import llm_gate

    original_gate = llm_gate.LLMGate
    valid_patch = (
        "FILE: tool_router.py\n"
        "<<<<<<< SEARCH\n"
        "old_value = 1\n"
        "=======\n"
        "old_value = 2\n"
        ">>>>>>> REPLACE"
    )

    calls: list[dict] = []

    class EmptyThenPatchGate:
        async def call_cloud(self, system_prompt, user_message, **kwargs):  # noqa: ANN001
            calls.append({
                "system_prompt": system_prompt,
                "user_message": user_message,
                **kwargs,
            })
            return "" if len(calls) == 1 else valid_patch

    try:
        llm_gate.LLMGate = EmptyThenPatchGate  # type: ignore[assignment]
        recovered = asyncio.run(
            su.generate_llm_patch(
                "修复一次瞬态空响应",
                ["tool_router.py"],
                ["empty_response"],
            )
        )
    finally:
        llm_gate.LLMGate = original_gate

    assert recovered["success"] is True, recovered
    assert len(recovered["edits"]) == 1, recovered
    assert len(calls) == su.PATCH_EMPTY_RESPONSE_ATTEMPTS == 2, calls
    assert all(call.get("profile_name") == "work" for call in calls), calls
    assert all(
        call.get("max_tokens_override") == su.PATCH_MAX_OUTPUT_TOKENS == 8192
        for call in calls
    ), calls

    empty_calls: list[dict] = []

    class AlwaysEmptyGate:
        async def call_cloud(self, system_prompt, user_message, **kwargs):  # noqa: ANN001
            empty_calls.append(kwargs)
            return ""

    try:
        llm_gate.LLMGate = AlwaysEmptyGate  # type: ignore[assignment]
        exhausted = asyncio.run(
            su.generate_llm_patch(
                "验证持续空响应有界退出",
                ["tool_router.py"],
                [],
            )
        )
    finally:
        llm_gate.LLMGate = original_gate

    assert exhausted["success"] is False, exhausted
    assert exhausted.get("declined") is False, exhausted
    assert exhausted["error"] == "LLM returned empty response", exhausted
    assert len(empty_calls) == su.PATCH_EMPTY_RESPONSE_ATTEMPTS == 2, empty_calls
    assert all(
        call.get("max_tokens_override") == su.PATCH_MAX_OUTPUT_TOKENS
        for call in empty_calls
    ), empty_calls


def _test_call_cloud_payload_budget() -> None:
    import llm_gate

    payloads: list[dict] = []
    original_client = llm_gate.httpx.AsyncClient

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"choices": [{"message": {"content": "ok"}}]}

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):  # noqa: ANN001
            return False

        async def post(self, url, *, headers, json):  # noqa: ANN001
            payloads.append(dict(json))
            return FakeResponse()

    try:
        llm_gate.httpx.AsyncClient = FakeAsyncClient  # type: ignore[assignment]
        gate = llm_gate.LLMGate()
        profile = gate.profiles["work"]
        profile.api_url = "https://example.invalid/v1"
        profile.api_key = "test-only-key"
        profile.model = "test-model"

        with_budget = asyncio.run(
            gate.call_cloud(
                "system",
                "user",
                profile_name="work",
                max_tokens_override=8192,
            )
        )
        default_budget = asyncio.run(
            gate.call_cloud("system", "user", profile_name="work")
        )
    finally:
        llm_gate.httpx.AsyncClient = original_client

    assert with_budget == "ok", with_budget
    assert default_budget == "ok", default_budget
    assert payloads[0].get("max_tokens") == 8192, payloads
    assert "max_tokens" not in payloads[1], payloads


def main() -> None:
    import evolution.self_upgrade as su

    _test_call_cloud_payload_budget()
    _test_internal_empty_response_retry(su)

    with tempfile.TemporaryDirectory() as tmp:
        proposal_file = Path(tmp) / "self_upgrade_proposals.jsonl"
        os.environ["SELF_UPGRADE_PROPOSALS_FILE"] = str(proposal_file)
        os.environ["EVOLUTION_UPGRADE_APPLY_ENABLED"] = "true"

        proposal = su.create_candidate_proposal(
            source="test_transient_retry",
            problem="提高工具成功率：主要失败类型=other_error",
            evidence=["tool_failure_count=27"],
            proposed_files=["tool_router.py"],
            patch_summary="为工具失败分类补充降级路径。",
        )
        assert proposal.status == "approved", proposal
        assert proposal.patch_attempt_count == 0, "new proposal should start at 0 attempts"
        su.append_proposal(proposal)

        # Force every patch generation to look like a transient empty-LLM failure.
        async def fake_empty_patch(problem, files, evidence):  # noqa: ANN001
            return {
                "success": False,
                "declined": False,
                "edits": [],
                "raw_output": "",
                "error": "LLM returned empty response",
            }

        su.generate_llm_patch = fake_empty_patch  # type: ignore[assignment]

        max_attempts = su.MAX_PATCH_ATTEMPTS
        assert max_attempts >= 2, "retry bound must allow at least one retry"

        # Attempts below the bound keep the proposal 'approved' for the next try.
        for expected in range(1, max_attempts):
            result = asyncio.run(su.generate_and_apply_patch(proposal.id))
            assert result["applied"] is False, result
            assert not result.get("blocked"), f"should not block before bound: {result}"
            latest = su.get_proposal(proposal.id)
            assert latest is not None
            assert latest["status"] == "approved", f"attempt {expected}: {latest['status']}"
            assert int(latest.get("patch_attempt_count") or 0) == expected, latest

        # The attempt that reaches the bound downgrades the proposal to 'blocked'.
        final = asyncio.run(su.generate_and_apply_patch(proposal.id))
        assert final["applied"] is False, final
        assert final.get("blocked") is True, f"final attempt should report blocked: {final}"
        latest = su.get_proposal(proposal.id)
        assert latest is not None
        assert latest["status"] == "blocked", f"expected blocked, got {latest['status']}"
        assert int(latest.get("patch_attempt_count") or 0) == max_attempts, latest

        # A blocked proposal is no longer eligible for apply, so the engine stops.
        after_block = asyncio.run(su.generate_and_apply_patch(proposal.id))
        assert after_block["applied"] is False, after_block
        assert "not approved" in str(after_block.get("reason", "")), after_block

    print("self_upgrade_transient_retry_bound: ok")


if __name__ == "__main__":
    main()
