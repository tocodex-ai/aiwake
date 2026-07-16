# -*- coding: utf-8 -*-
"""Self-check: file_read 分页续读能把大文件读全，支撑 AIwake 自我迭代闭环。

验证 _file_read 的 offset/limit 分页语义：
- 截断时返回 truncated=True / total_chars / next_offset；
- 用 next_offset 续读能拿到后续内容，且全程拼接可还原完整文件；
- 末段 truncated=False、next_offset=None；
- 非法/越界参数有安全兜底，不抛异常。

用一个可控的临时文件（长度可一次性读全）做基准，避免依赖被钳制上限截断的真实大文件。
"""
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch


def _find_agent_dir() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "tool_router.py").exists():
            return parent
    raise RuntimeError("找不到 agent-mind 目录（tool_router.py）")


AGENT_DIR = _find_agent_dir()
sys.path.insert(0, str(AGENT_DIR))

from tool_router import ToolRouter  # noqa: E402


def main() -> None:
    router = ToolRouter()
    # 构造一个内容已知、不含敏感模式、长度可一次性读全(<40000)的临时文件
    lines = [f"line-{i:05d}-abcdefghijklmnopqrstuvwxyz" for i in range(800)]
    body = "\n".join(lines)  # 约 3.4 万字符
    fd, tmp_path = tempfile.mkstemp(suffix=".txt", prefix="aiwake_pgtest_", dir=str(AGENT_DIR))
    os.close(fd)
    Path(tmp_path).write_text(body, encoding="utf-8")
    try:
        _run_assertions(router, tmp_path, body)
    except AssertionError as e:
        print(f"[FAIL] {e}")
        sys.exit(1)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
    print("[PASS] file_read 分页续读自检全部通过")


def _strip_notes(seg: str, offset: int, truncated: bool) -> str:
    """从返回窗口剥离续读头注与截断尾注，还原真实正文片段。"""
    body = seg
    if truncated:
        body = body.split("\n...[内容已截断", 1)[0]
    if offset > 0 and body.startswith("...[从第"):
        body = body.split("]\n", 1)[1]
    return body


def _run_assertions(router: ToolRouter, tmp_path: str, expected: str) -> None:
    total = len(expected)
    assert total > 8000, f"测试基准文件应足够大用于分页验证: total={total}"
    assert total < 40000, f"测试基准文件应能一次性读全: total={total}"

    # 0. 一次性读全作基准（limit 足够大）
    full = router._file_read(tmp_path, 0, 40000)
    assert full.get("status") == "ok", f"基准读取失败: {full}"
    assert full.get("truncated") is False, f"基准应未截断: {full.get('truncated')}"
    assert full.get("total_chars") == total, f"total_chars 应={total}, got {full.get('total_chars')}"
    assert full.get("result") == expected, "一次性读取内容应与原文件一致"

    # 1. 首段：小 limit 必然触发截断，返回分页元信息
    page1 = router._file_read(tmp_path, 0, 2000)
    assert page1.get("status") == "ok", f"首段读取失败: {page1}"
    assert page1.get("truncated") is True, f"首段应被截断: {page1.get('truncated')}"
    assert page1.get("total_chars") == total, "total_chars 应与基准一致"
    assert page1.get("next_offset") == 2000, f"next_offset 应为 2000: {page1.get('next_offset')}"
    assert "[内容已截断" in page1.get("result", ""), "截断段应含可操作的续读提示"

    # 2. 续读：用 next_offset 拼接，应能逐段还原完整文件正文
    reconstructed = ""
    offset = 0
    guard = 0
    while True:
        guard += 1
        assert guard < 100, "分页续读循环次数异常，可能未收敛"
        page = router._file_read(tmp_path, offset, 5000)
        assert page.get("status") == "ok", f"续读失败 offset={offset}: {page}"
        reconstructed += _strip_notes(page.get("result", ""), offset, bool(page.get("truncated")))
        if not page.get("truncated"):
            assert page.get("next_offset") is None, f"末段 next_offset 应为 None: {page}"
            break
        offset = page.get("next_offset")
    assert reconstructed == expected, "分页拼接结果应与一次性读取完全一致"

    # 3. offset 越界：返回空窗口、不截断、不抛异常
    over = router._file_read(tmp_path, total + 100, 1000)
    assert over.get("status") == "ok", f"越界读取应安全返回: {over}"
    assert over.get("truncated") is False, "越界不应标记截断"
    assert over.get("next_offset") is None, "越界 next_offset 应为 None"

    # 4. 非法参数：负 offset / 非数字 limit 应安全兜底
    bad = router._file_read(tmp_path, -5, "oops")  # type: ignore[arg-type]
    assert bad.get("status") == "ok", f"非法参数应安全兜底: {bad}"
    assert bad.get("offset") == 0, "负 offset 应被钳制为 0"
    assert isinstance(bad.get("limit"), int) and bad.get("limit") >= 1, "非法 limit 应回退到合法默认值"

    # 5. limit 上限钳制到 40000
    huge = router._file_read(tmp_path, 0, 999999)
    assert huge.get("limit") == 40000, f"limit 应被钳制到 40000: {huge.get('limit')}"


if __name__ == "__main__":
    main()
