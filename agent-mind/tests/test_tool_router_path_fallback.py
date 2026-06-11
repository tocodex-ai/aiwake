"""Self-check: file_read/file_list 前缀回退能命中容器根目录下的自身源码。

容器内 WORKDIR=/app 且 COPY . . 将 src/agent-mind/ 内容铺平到 /app 根，
因此 AIwake 用仓库相对路径 "src/agent-mind/tool_router.py" 内省自身代码时，
应能通过去掉 src/agent-mind/ 前缀并在 APP_ROOT 下重试而命中文件。

Run from repository root:
    python src/agent-mind/tests/test_tool_router_path_fallback.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch


def _find_agent_dir() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
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

        # 3. file_list：仓库相对目录应能通过去前缀回退列举
        res3 = router._file_list("src/agent-mind/evolution")
        assert res3.get("status") == "ok", f"expected ok for dir list, got {res3}"
        names = {item["name"] for item in res3.get("result", [])}
        assert "self_upgrade.py" in names, f"self_upgrade.py not in listing: {names}"

        # 4. 不存在的文件仍应明确报错（不误命中）
        res4 = router._file_read("src/agent-mind/does_not_exist_xyz.py")
        assert res4.get("diagnosis") == "file_not_found", f"expected file_not_found, got {res4}"

    print("tool_router_path_fallback_selfcheck: ok")


if __name__ == "__main__":
    main()
