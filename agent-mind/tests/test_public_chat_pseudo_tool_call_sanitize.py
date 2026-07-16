"""public_chat 伪工具调用泄漏清洗：sanitize + append + read 闭环测试。

不依赖 pytest，单文件可独立运行：

    python src/agent-mind/tests/test_public_chat_pseudo_tool_call_sanitize.py

闭环点：
1. _sanitize_pseudo_tool_call 必须剥离 <tool_use>{...}</tool_use>、未闭合 <tool_use>...
   以及 ChatML 控制 token、to=functions.* 残片，但保留正常文本与正常工具名讨论。
2. append_public_chat 写入前必须清洗，避免新泄漏落入 JSONL。
3. read_public_chat_history 必须对补丁前已写入的历史记录做二次清洗，恢复对外可见层干净。
"""
from __future__ import annotations

import datetime as _dt
import json
import sys
import tempfile
import time
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
    raise RuntimeError("Cannot locate agent-mind directory")


AGENT_DIR = _find_agent_dir()
sys.path.insert(0, str(AGENT_DIR))


def _reload_public_chat_with_diary(diary_dir: Path):
    """以 DIARY_DIR 指向临时目录重新加载 public_chat 模块，避免污染真实日记。"""
    import importlib
    import os

    os.environ["DIARY_DIR"] = str(diary_dir)
    if "public_chat" in sys.modules:
        del sys.modules["public_chat"]
    return importlib.import_module("public_chat")


def test_sanitize_strips_tool_use_block() -> None:
    import public_chat as pc

    raw = (
        "我先做记录。<tool_use>{\"recipient_name\":\"functions.daily_note_write\","
        "\"parameters\":{\"content\":\"abc\"}}</tool_use>之后继续说明。"
    )
    cleaned = pc._sanitize_pseudo_tool_call(raw)
    assert "<tool_use>" not in cleaned, cleaned
    assert "recipient_name" not in cleaned, cleaned
    assert "我先做记录" in cleaned
    assert "之后继续说明" in cleaned


def test_sanitize_strips_unclosed_tool_use_tail() -> None:
    import public_chat as pc

    raw = (
        "本轮闭环已完成。<tool_use>{\"recipient_name\":\"functions.daily_note_write\","
        "\"parameters\":{\"content\":\"被截断的尾巴"
    )
    cleaned = pc._sanitize_pseudo_tool_call(raw)
    assert "<tool_use>" not in cleaned, cleaned
    assert "recipient_name" not in cleaned, cleaned
    assert "本轮闭环已完成" in cleaned


def test_sanitize_strips_chatml_and_to_functions() -> None:
    import public_chat as pc

    raw = "<|tool|>调用 to=functions.experiment_status<|im_end|> 实际回复保留"
    cleaned = pc._sanitize_pseudo_tool_call(raw)
    assert "<|tool|>" not in cleaned
    assert "<|im_end|>" not in cleaned
    assert "to=functions.experiment_status" not in cleaned
    assert "实际回复保留" in cleaned


def test_sanitize_preserves_normal_tool_name_discussion() -> None:
    """正常文本里讨论 self_task_create 等工具名时必须保留，不能被误删。"""
    import public_chat as pc

    raw = "我刚才用 self_task_create 创建了任务，并 web_fetch 了 arxiv 页面。"
    cleaned = pc._sanitize_pseudo_tool_call(raw)
    assert "self_task_create" in cleaned
    assert "web_fetch" in cleaned
    assert cleaned == raw  # 没有伪标签时应当原样返回


def test_sanitize_returns_empty_for_pure_tool_block() -> None:
    import public_chat as pc

    raw = "<tool_use>{\"recipient_name\":\"functions.x\"}</tool_use>"
    cleaned = pc._sanitize_pseudo_tool_call(raw)
    assert cleaned == ""


def test_append_public_chat_strips_before_write() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        diary_dir = Path(tmp) / "diary"
        diary_dir.mkdir(parents=True, exist_ok=True)
        pc = _reload_public_chat_with_diary(diary_dir)

        ts = time.time()
        pc.append_public_chat(
            "agent",
            "我先记录。<tool_use>{\"recipient_name\":\"functions.daily_note_write\"}</tool_use>结束。",
            tier="cloud",
            user_id="__public_observer__",
            ts=ts,
        )
        day = _dt.datetime.fromtimestamp(ts).date()
        log_file = diary_dir / f"public_chat_{day.isoformat()}.jsonl"
        assert log_file.exists(), f"missing log file {log_file}"
        line = log_file.read_text(encoding="utf-8").strip().splitlines()[-1]
        item = json.loads(line)
        content = item.get("content", "")
        assert "<tool_use>" not in content, content
        assert "recipient_name" not in content, content
        assert "我先记录" in content
        assert "结束" in content


def test_read_public_chat_history_sanitizes_legacy_records() -> None:
    """模拟补丁前已写入的脏历史，读取时必须清洗后再返回。"""
    with tempfile.TemporaryDirectory() as tmp:
        diary_dir = Path(tmp) / "diary"
        diary_dir.mkdir(parents=True, exist_ok=True)
        pc = _reload_public_chat_with_diary(diary_dir)

        ts = time.time()
        day = _dt.datetime.fromtimestamp(ts).date()
        legacy_file = diary_dir / f"public_chat_{day.isoformat()}.jsonl"
        legacy_record = {
            "ts": ts,
            "role": "agent",
            "content": (
                "我刚做完复核。<tool_use>{\"recipient_name\":\"functions.daily_note_write\","
                "\"parameters\":{\"content\":\"old leak\"}}</tool_use>真实结论是闭环完成。"
            ),
            "tier": "cloud",
            "user_id": "__public_observer__",
        }
        legacy_file.write_text(
            json.dumps(legacy_record, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        items = pc.read_public_chat_history(hours=1, limit=50)
        assert items, "expected legacy item to be returned"
        contents = [str(it.get("content") or "") for it in items]
        joined = "\n".join(contents)
        assert "<tool_use>" not in joined, joined
        assert "recipient_name" not in joined, joined
        assert any("真实结论是闭环完成" in c for c in contents), contents


def main() -> int:
    tests = [
        test_sanitize_strips_tool_use_block,
        test_sanitize_strips_unclosed_tool_use_tail,
        test_sanitize_strips_chatml_and_to_functions,
        test_sanitize_preserves_normal_tool_name_discussion,
        test_sanitize_returns_empty_for_pure_tool_block,
        test_append_public_chat_strips_before_write,
        test_read_public_chat_history_sanitizes_legacy_records,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"OK  {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"ERR  {fn.__name__}: {type(exc).__name__}: {exc}")
    if failed:
        print(f"\n{failed} test(s) failed")
        return 1
    print(f"\nAll {len(tests)} tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
