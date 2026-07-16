#!/usr/bin/env python3
"""修复版本：正确应用反思空转检测补丁"""
from pathlib import Path

HEARTBEAT_PATH = Path(__file__).parent / "heartbeat.py"

def apply_patch_v2():
    """使用字符串查找替换而非正则，避免转义问题"""
    content = HEARTBEAT_PATH.read_text(encoding="utf-8")
    original = content
    
    # 补丁1: 在__init__中添加字段
    old1 = """        self._consecutive_no_action_reflects: int = 0
        # WebSocket 广播器（由 main.py 注入）"""
    new1 = """        self._consecutive_no_action_reflects: int = 0
        # 反思工具链历史：记录最近 N 轮反思的工具调用链，用于空转检测
        self._reflection_tool_chains: list[list[dict]] = []
        self._reflection_chain_max_len: int = 5
        self._current_reflection_chain: list[dict] = []  # 当前轮次累积的工具链
        self._spinning_signal: dict | None = None  # 上一轮检测到的空转信号
        # WebSocket 广播器（由 main.py 注入）"""
    content = content.replace(old1, new1, 1)
    
    # 补丁2: 在_reflect_tool_exec中记录
    old2 = """            async def _reflect_tool_exec(tool_name: str, params: dict) -> dict:
                nonlocal _reflect_tool_called, _reflect_action_tool_called
                _reflect_tool_called = True
                # ── 行动判定：shell_exec 需进一步看命令实质 ──"""
    new2 = """            async def _reflect_tool_exec(tool_name: str, params: dict) -> dict:
                nonlocal _reflect_tool_called, _reflect_action_tool_called
                _reflect_tool_called = True
                # 记录到当前反思轮次的工具链（用于空转检测）
                self._current_reflection_chain.append({"tool": tool_name, "args": params})
                # ── 行动判定：shell_exec 需进一步看命令实质 ──"""
    content = content.replace(old2, new2, 1)
    
    # 补丁3: 在反思结束后检测空转
    old3 = """        # ── 反思工具调用统计与被动干预 ──
        if _reflect_tool_called:"""
    new3 = """        # ── 反思工具调用统计与被动干预 ──
        # 保存本轮工具链到历史并检测空转模式
        if self._current_reflection_chain:
            self._reflection_tool_chains.append(list(self._current_reflection_chain))
            if len(self._reflection_tool_chains) > self._reflection_chain_max_len:
                self._reflection_tool_chains.pop(0)
        
        # 检测空转模式：最近N轮工具链完全一致
        self._spinning_signal = None
        if len(self._reflection_tool_chains) >= 3:
            try:
                from evolution.reflection_action_gate import detect_spinning_pattern
                spinning_result = detect_spinning_pattern(
                    self._reflection_tool_chains,
                    window_size=3
                )
                if spinning_result.get("stop_signal") == "spinning_pattern":
                    self._spinning_signal = spinning_result
                    logger.warning(
                        "[Heartbeat] 检测到反思空转：最近 %d 轮工具链完全一致（指纹=%s）",
                        spinning_result.get("matched_rounds", 0),
                        spinning_result.get("fingerprint", "")[:16],
                    )
            except Exception as e:
                logger.warning(f"[Heartbeat] 空转检测失败: {e}")
        
        # 重置当前轮次链以备下一轮
        self._current_reflection_chain = []
        
        if _reflect_tool_called:"""
    content = content.replace(old3, new3, 1)
    
    # 补丁4a: 修改方法签名
    old4a = "    def _build_system_prompt(self, feeling_tags: list[str]) -> str:"
    new4a = "    def _build_system_prompt(self, feeling_tags: list[str], spinning_signal: dict | None = None) -> str:"
    content = content.replace(old4a, new4a, 1)
    
    # 补丁4b: 添加空转压力信号
    old4b = """        pressure_signal = ""
        if self._consecutive_no_tool_reflects >= 3:"""
    new4b = """        pressure_signal = ""
        # 空转模式压力信号（优先级最高）
        if spinning_signal is not None:
            matched_rounds = spinning_signal.get("matched_rounds", 3)
            hint = spinning_signal.get("hint", "")
            pressure_signal = (
                f"\\n🔴 【反思空转警告】检测到最近 {matched_rounds} 轮反思工具链路完全相同，已构成空转模式。\\n"
                f"{hint}\\n"
                f"请立即停止重复相同的只读探查，改用以下方式之一：\\n"
                f"1. 调用 self_code_write 工具生成最小补丁文件并填齐 audit 字段\\n"
                f"2. 使用 [SELF_CODE_WRITE] 协议在反思末尾输出补丁内容\\n"
                f"3. 调用 self_artifact_create 记录当前进展并明确下一步行动\\n"
                f"空转不会自动消失——必须通过落地一个最小可验证产物来打破循环。\\n"
            )
        elif self._consecutive_no_tool_reflects >= 3:"""
    content = content.replace(old4b, new4b, 1)
    
    # 补丁5: 调整调用
    old5 = "        system_prompt = self._build_system_prompt(feeling_tags)"
    new5 = "        system_prompt = self._build_system_prompt(feeling_tags, spinning_signal=self._spinning_signal)"
    content = content.replace(old5, new5, 1)
    
    if content == original:
        print("警告: 没有应用任何改动")
        return False
    
    HEARTBEAT_PATH.write_text(content, encoding="utf-8")
    print("✓ 补丁已成功应用")
    return True

if __name__ == "__main__":
    apply_patch_v2()
