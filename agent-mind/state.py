"""
AIwake - 内在状态机
基于 AIwake runtime principles的工程化实现
"""
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

STATE_FILE = Path(__file__).parent / "data" / "state.json"


@dataclass
class InternalState:
    # 欲望引擎三大向量 (0.0 ~ 1.0)
    TR: float = 0.6   # 兴奋/奖励
    CS: float = 0.7   # 满足/安全
    SA: float = 0.2   # 压力/警觉

    # 仪表盘标签
    energy_level: str = "精力充沛"
    mood_level: str = "心情愉快"
    patience_level: str = "很有耐心"
    mental_fatigue: float = 0.0

    # 当前主动目标
    active_goal: str = "维持稳定状态，等待交互"

    # 上次更新时间戳
    last_updated: float = field(default_factory=time.time)

    # 心跳计数
    tick_count: int = 0

    def update_vectors(self, tr_delta: float = 0, cs_delta: float = 0, sa_delta: float = 0):
        """更新向量并自然衰减"""
        self.TR = max(0.0, min(1.0, self.TR + tr_delta))
        self.CS = max(0.0, min(1.0, self.CS + cs_delta))
        self.SA = max(0.0, min(1.0, self.SA + sa_delta))
        self._update_dashboard()
        self.last_updated = time.time()

    def natural_decay(self):
        """自然衰减：向健康平衡点回归"""
        TARGET_TR, TARGET_CS, TARGET_SA = 0.6, 0.7, 0.2
        DECAY = 0.02
        self.TR += (TARGET_TR - self.TR) * DECAY
        self.CS += (TARGET_CS - self.CS) * DECAY
        self.SA += (TARGET_SA - self.SA) * DECAY
        self.mental_fatigue = max(0.0, self.mental_fatigue - 0.01)
        self._update_dashboard()

    def _update_dashboard(self):
        """根据向量更新仪表盘标签"""
        # 精力
        energy = self.TR * 0.4 + self.CS * 0.4 - self.SA * 0.3 - self.mental_fatigue * 0.3
        if energy > 0.7:
            self.energy_level = "精力充沛"
        elif energy > 0.4:
            self.energy_level = "精力尚可"
        else:
            self.energy_level = "精疲力竭"

        # 心情
        mood = self.TR * 0.5 + self.CS * 0.5 - self.SA * 0.4
        if mood > 0.7:
            self.mood_level = "心情愉快"
        elif mood > 0.4:
            self.mood_level = "心情平静"
        else:
            self.mood_level = "心情低落"

        # 耐心
        patience = self.CS * 0.6 - self.SA * 0.5 - self.mental_fatigue * 0.3
        if patience > 0.5:
            self.patience_level = "很有耐心"
        elif patience > 0.2:
            self.patience_level = "耐心有限"
        else:
            self.patience_level = "耐心耗尽"

    def get_feeling_tags(self) -> list[str]:
        """返回当前感受标签，供 LLM 使用"""
        tags = [self.energy_level, self.mood_level, self.patience_level]
        if self.SA > 0.7:
            tags.append("感到焦虑和紧绷")
        if self.TR > 0.8:
            tags.append("充满探索欲望")
        if self.CS < 0.3:
            tags.append("感到不安全，需要稳定")
        if self.mental_fatigue > 0.7:
            tags.append("大脑感觉转不动了")
        return tags

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self):
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2))

    @classmethod
    def load(cls) -> "InternalState":
        if STATE_FILE.exists():
            try:
                data = json.loads(STATE_FILE.read_text())
                return cls(**data)
            except Exception:
                pass
        return cls()
