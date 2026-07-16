#!/usr/bin/env python3
"""
自动为 heartbeat.py 添加反思空转检测功能
用法: python apply_spinning_detection.py
"""
import re
from pathlib import Path

HEARTBEAT_PATH = Path(__file__).parent / "heartbeat.py"

def apply_patch():
    """应用反思空转检测补丁"""
    if not HEARTBEAT_PATH.exists():
        print(f"错误: 找不到文件 {HEARTBEAT_PATH}")
        return False
    
    content = HEARTBEAT_PATH.read_text(encoding="utf-8")
    original_content = content
    
    # ========== 补丁 1: 在 __init__ 中添加工具链历史记录字段 ==========
    pattern1 = r"(self\._consecutive_no_action_reflects: int = 0)\n(\s+# WebSocket)"
    replacement1 = (
        r"\1\n"
        r"        # 反思工具链历史：记录最近 N 轮反思的工具调用链，用于空转检测\n"
        r"        self._reflection_tool_chains: list[list[dict]] = []\n"
        r"        self._reflection_chain_max_len: int = 5\n"
        r"        self._current_reflection_chain: list[dict] = []  # 当前轮次累积的工具链\n"
        r"        self._spinning_signal: dict | None = None  # 上一轮检测到的空转信号\n"
        r"\2"
    )
    content = re.sub(pattern1, replacement1, content, count=1)
    
    # ========== 补丁 2: 在 _reflect_tool_exec 中记录工具调用 ==========
    pattern2 = r"(async def _reflect_tool_exec\(tool_name: str, params: dict\) -> dict:\n\s+nonlocal _reflect_tool_called, _reflect_action_tool_called\n\s+_reflect_tool_called = True)"
    replacement2 = (
        r"\1\n"
        r"                # 记录到当前反思轮次的工具链（用于空转检测）\n"
        r"                self._current_reflection_chain.append({\"tool\": tool_name, \"args\": params})"
    )
    content = re.sub(pattern2, replacement2, content, count=1)
    
    # ========== 补丁 3: 在反思结束后检测空转 ==========
    # 找到 "# ── 反思工具调用统计与被动干预 ──" 之后、"if _reflect_tool_called:" 之前插入
    pattern3 = r"(# ── 反思工具调用统计与被动干预 ──\n)(\s+)(if _reflect_tool_called:)"
    replacement3 = (
        r"\1"
        r"\2# 保存本轮工具链到历史并检测空转模式\n"
        r"\2if self._current_reflection_chain:\n"
        r"\2    self._reflection_tool_chains.append(list(self._current_reflection_chain))\n"
        r"\2    if len(self._reflection_tool_chains) > self._reflection_chain_max_len:\n"
        r"\2        self._reflection_tool_chains.pop(0)\n"
        r"\2\n"
        r"\2# 检测空转模式：最近N轮工具链完全一致\n"
        r"\2self._spinning_signal = None\n"
        r"\2if len(self._reflection_tool_chains) >= 3:\n"
        r"\2    try:\n"
        r"\2        from evolution.reflection_action_gate import detect_spinning_pattern\n"
        r"\2        spinning_result = detect_spinning_pattern(\n"
        r"\2            self._reflection_tool_chains,\n"
        r"\2            window_size=3\n"
        r"\2        )\n"
        r"\2        if spinning_result.get(\"stop_signal\") == \"spinning_pattern\":\n"
        r"\2            self._spinning_signal = spinning_result\n"
        r"\2            logger.warning(\n"
        r"\2                \"[Heartbeat] 检测到反思空转：最近 %d 轮工具链完全一致（指纹=%s）\",\n"
        r"\2                spinning_result.get(\"matched_rounds\", 0),\n"
        r"\2                spinning_result.get(\"fingerprint\", \"\")[:16],\n"
        r"\2            )\n"
        r"\2    except Exception as e:\n"
        r"\2        logger.warning(f\"[Heartbeat] 空转检测失败: {e}\")\n"
        r"\2\n"
        r"\2# 重置当前轮次链以备下一轮\n"
        r"\2self._current_reflection_chain = []\n"
        r"\2\n"
        r"\2\3"
    )
    content = re.sub(pattern3, replacement3, content, count=1)
    
    # ========== 补丁 4: 修改 _build_system_prompt 方法签名和内容 ==========
    # 4.1 修改方法签名
    pattern4a = r"def _build_system_prompt\(self, feeling_tags: list\[str\]\) -> str:"
    replacement4a = r"def _build_system_prompt(self, feeling_tags: list[str], spinning_signal: dict | None = None) -> str:"
    content = re.sub(pattern4a, replacement4a, content, count=1)
    
    # 4.2 在压力信号部分添加空转检测（在 "pressure_signal = ""\" 之后）
    pattern4b = r"(pressure_signal = \"\")\n(\s+)(if self\._consecutive_no_tool_reflects >= 3:)"
    replacement4b = (
        r"\1\n"
        r"\2# 空转模式压力信号（优先级最高）\n"
        r"\2if spinning_signal is not None:\n"
        r"\2    matched_rounds = spinning_signal.get(\"matched_rounds\", 3)\n"
        r"\2    hint = spinning_signal.get(\"hint\", \"\")\n"
        r"\2    pressure_signal = (\n"
        r"\2        f\"\\n🔴 【反思空转警告】检测到最近 {matched_rounds} 轮反思工具链路完全相同，已构成空转模式。\\n\"\n"
        r"\2        f\"{hint}\\n\"\n"
        r"\2        f\"请立即停止重复相同的只读探查，改用以下方式之一：\\n\"\n"
        r"\2        f\"1. 调用 self_code_write 工具生成最小补丁文件并填齐 audit 字段\\n\"\n"
        r"\2        f\"2. 使用 [SELF_CODE_WRITE] 协议在反思末尾输出补丁内容\\n\"\n"
        r"\2        f\"3. 调用 self_artifact_create 记录当前进展并明确下一步行动\\n\"\n"
        r"\2        f\"空转不会自动消失——必须通过落地一个最小可验证产物来打破循环。\\n\"\n"
        r"\2    )\n"
        r"\2el\3"
    )
    content = re.sub(pattern4b, replacement4b, content, count=1)
    
    # ========== 补丁 5: 调整 _do_reflection 中调用 _build_system_prompt 传入 spinning_signal ==========
    pattern5 = r"system_prompt = self\._build_system_prompt\(feeling_tags\)"
    replacement5 = r"system_prompt = self._build_system_prompt(feeling_tags, spinning_signal=self._spinning_signal)"
    content = re.sub(pattern5, replacement5, content, count=1)
    
    # 检查是否有改动
    if content == original_content:
        print("警告: 没有应用任何改动，可能补丁已经应用过或代码结构已变化")
        return False
    
    # 备份原文件
    backup_path = HEARTBEAT_PATH.with_suffix(".py.backup")
    backup_path.write_text(original_content, encoding="utf-8")
    print(f"已备份原文件到: {backup_path}")
    
    # 写入修改后的内容
    HEARTBEAT_PATH.write_text(content, encoding="utf-8")
    print(f"✅ 补丁已成功应用到: {HEARTBEAT_PATH}")
    print("\n改动总结:")
    print("  1. 在 __init__ 中添加了工具链历史记录字段")
    print("  2. 在 _reflect_tool_exec 中记录每个工具调用")
    print("  3. 在反思结束后调用 detect_spinning_pattern 检测空转")
    print("  4. 修改 _build_system_prompt 接收 spinning_signal 参数")
    print("  5. 在 system prompt 中注入空转压力信号")
    print("  6. 调整 _do_reflection 传入 spinning_signal")
    return True

if __name__ == "__main__":
    success = apply_patch()
    if success:
        print("\n下一步: 运行 pytest 验证改动，然后部署到 Fly.io")
    else:
        print("\n补丁应用失败，请手动检查")
