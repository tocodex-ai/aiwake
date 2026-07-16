# -*- coding: utf-8 -*-
"""web_fetch 自我观察通道测试：

1. 抓取自身线上域名的只读端点时应改写为容器内部地址，不再被 SSRF 防护误拒。
2. 抓取自身域名的非只读端点时不改写（仍走常规 SSRF 检查）。
3. 其他私网/回环地址仍然被拦截。
"""
import asyncio
import sys
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tool_router import ToolRouter


def _router() -> ToolRouter:
    return ToolRouter.__new__(ToolRouter)  # 跳过 __init__，只测纯逻辑方法


def test_rewrite_self_url_readonly_endpoint():
    router = _router()
    parsed = urlparse("https://aiwake.fly.dev/evolution/status")
    rewritten = router._rewrite_self_url(parsed)
    assert rewritten == "http://127.0.0.1:8000/evolution/status", rewritten


def test_rewrite_self_url_with_query():
    router = _router()
    parsed = urlparse("https://aiwake.fly.dev/activity_logs?hours=4&limit=50")
    rewritten = router._rewrite_self_url(parsed)
    assert rewritten == "http://127.0.0.1:8000/activity_logs?hours=4&limit=50", rewritten


def test_rewrite_self_url_root_allowed():
    router = _router()
    parsed = urlparse("https://aiwake.fly.dev/")
    rewritten = router._rewrite_self_url(parsed)
    assert rewritten == "http://127.0.0.1:8000/", rewritten


def test_rewrite_self_url_rejects_non_readonly_path():
    router = _router()
    parsed = urlparse("https://aiwake.fly.dev/evolution/run")
    assert router._rewrite_self_url(parsed) is None


def test_rewrite_self_url_ignores_other_hosts():
    router = _router()
    parsed = urlparse("https://example.com/evolution/status")
    assert router._rewrite_self_url(parsed) is None


def test_private_address_still_blocked():
    router = _router()
    ok, note = asyncio.run(
        router._resolve_public_ips_for_fetch("localhost")
    )
    assert not ok
    assert "非公网地址" in note or "DNS" in note, note


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                failures += 1
                print(f"FAIL {name}: {e}")
    sys.exit(1 if failures else 0)
