"""Self-check: file_read/file_list 前缀回退能命中容器根目录下的自身源码。

容器内 WORKDIR=/app 且 COPY . . 将 src/agent-mind/ 内容铺平到 /app 根，
因此 AIwake 用仓库相对路径 "src/agent-mind/tool_router.py" 内省自身代码时，
应能通过去掉 src/agent-mind/ 前缀并在 APP_ROOT 下重试而命中文件。

R51 追加：带幻觉前缀的绝对路径（如 /app/src/agent-mind/evolution/self_upgrade.py，
线上反思真实出现过的 file_not_found 最大失败类别）也应能命中真实文件。

Run from repository root:
    python src/agent-mind/tests/test_tool_router_path_fallback.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch


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

from tool_router import ToolRouter  # noqa: E402


def main() -> None:
    router = ToolRouter()

    # 切换到中性临时目录运行，确保相对路径不会因当前工作目录恰好是仓库根而被
    # "direct" 命中，从而真正触发 src/agent-mind/ 去前缀回退逻辑（模拟容器 /app 环境）。
    original_cwd = os.getcwd()
    tmp = tempfile.mkdtemp(prefix="tr_path_fallback_")
    os.chdir(tmp)
    try:
        _run_assertions(router)
    finally:
        os.chdir(original_cwd)

    print("tool_router_path_fallback_selfcheck: ok")


def _run_assertions(router: ToolRouter) -> None:
    # 用本地仓库的 agent-mind 目录模拟容器内的 /app 根目录
    with patch.dict(os.environ, {"APP_ROOT": str(AGENT_DIR)}):
        # 1. file_read：仓库相对路径应能通过去前缀回退命中真实文件
        res = router._file_read("src/agent-mind/tool_router.py")
        assert res.get("status") == "ok", f"expected ok, got {res}"
        assert res.get("resolved_from") == "app_root_strip", f"unexpected resolved_from: {res.get('resolved_from')}"
        assert "Tool Router" in res.get("result", ""), "file content should contain module docstring"

        # 2. file_read：嵌套子目录路径同样应命中
        res2 = router._file_read("src/agent-mind/evolution/self_upgrade.py")
        assert res2.get("status") == "ok", f"expected ok for nested path, got {res2}"
        assert res2.get("resolved_from") == "app_root_strip"

        # 2b. R51 关键回归：带幻觉前缀的绝对路径（线上真实失败样本）也应命中。
        #     /app/src/agent-mind/evolution/self_upgrade.py 中 src/agent-mind/ 之后部分
        #     为 evolution/self_upgrade.py，应在 APP_ROOT 下命中。
        res2b = router._file_read("/app/src/agent-mind/evolution/self_upgrade.py")
        assert res2b.get("status") == "ok", f"expected ok for phantom-abs path, got {res2b}"
        assert res2b.get("resolved_from") == "app_root_strip"

        # 3. file_list：仓库相对目录应能通过去前缀回退列举
        res3 = router._file_list("src/agent-mind/evolution")
        assert res3.get("status") == "ok", f"expected ok for dir list, got {res3}"
        names = {item["name"] for item in res3.get("result", [])}
        assert "self_upgrade.py" in names, f"self_upgrade.py not in listing: {names}"

        # 3b. file_list：带幻觉前缀的绝对目录也应命中
        res3b = router._file_list("/app/src/agent-mind/evolution")
        assert res3b.get("status") == "ok", f"expected ok for phantom-abs dir, got {res3b}"
        names_b = {item["name"] for item in res3b.get("result", [])}
        assert "self_upgrade.py" in names_b, f"self_upgrade.py not in phantom-abs listing: {names_b}"

        # 4. 不存在的文件仍应明确报错（不误命中）
        res4 = router._file_read("src/agent-mind/does_not_exist_xyz.py")
        assert res4.get("diagnosis") == "file_not_found", f"expected file_not_found, got {res4}"
        assert "recommended_tools" not in res4, f"普通缺失文件不应附带工具推荐: {res4}"

        # 5. 幻觉绝对路径读结构化状态内容 → file_not_found 时应附带 recommended_tools
        res5 = router._file_read("/home/aiwake/aiwake/proposals/self_upgrade_status.md")
        assert res5.get("diagnosis") == "file_not_found", f"expected file_not_found, got {res5}"
        assert res5.get("recommended_tools") == ["self_upgrade_status"], f"应推荐 self_upgrade_status: {res5}"

        # 6. experiment_status 同理（含连字符/大小写归一化）
        res6 = router._file_read("data/Experiment-Status.json")
        assert res6.get("diagnosis") == "file_not_found", f"expected file_not_found, got {res6}"
        assert res6.get("recommended_tools") == ["experiment_status"], f"应推荐 experiment_status: {res6}"


if __name__ == "__main__":
    main()
