"""统一实验自主配置。

默认保持保守：实验全自主关闭、生产部署关闭、破坏性操作关闭。
所有调用方都应通过 load_autonomy_config() 读取当前环境变量，避免在进程启动后
环境切换时拿到陈旧配置。
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any


_TRUE_VALUES = {"1", "true", "yes", "on", "enabled", "y"}
_FALSE_VALUES = {"0", "false", "no", "off", "disabled", "n"}


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    return default


@dataclass(frozen=True)
class AutonomyConfig:
    experiment_full_autonomy: bool = False
    experiment_profile: str = "vm"
    experiment_allow_shell: bool = True
    experiment_allow_file_write: bool = True
    experiment_allow_network: bool = True
    experiment_allow_patch_generation: bool = True
    experiment_allow_self_tasks: bool = True
    experiment_allow_deploy: bool = False
    experiment_allow_destructive: bool = False
    upgrade_proposal_enabled: bool = True
    upgrade_auto_validate: bool = True
    upgrade_human_approval_required: bool = False
    upgrade_apply_enabled: bool = True
    upgrade_deploy_enabled: bool = True
    upgrade_max_files_per_proposal: int = 3

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_autonomy_config() -> AutonomyConfig:
    return AutonomyConfig(
        experiment_full_autonomy=_env_bool("EXPERIMENT_FULL_AUTONOMY", False),
        experiment_profile=os.getenv("EXPERIMENT_PROFILE", "vm").strip() or "vm",
        experiment_allow_shell=_env_bool("EXPERIMENT_ALLOW_SHELL", True),
        experiment_allow_file_write=_env_bool("EXPERIMENT_ALLOW_FILE_WRITE", True),
        experiment_allow_network=_env_bool("EXPERIMENT_ALLOW_NETWORK", True),
        experiment_allow_patch_generation=_env_bool("EXPERIMENT_ALLOW_PATCH_GENERATION", True),
        experiment_allow_self_tasks=_env_bool("EXPERIMENT_ALLOW_SELF_TASKS", True),
        experiment_allow_deploy=_env_bool("EXPERIMENT_ALLOW_DEPLOY", False),
        experiment_allow_destructive=_env_bool("EXPERIMENT_ALLOW_DESTRUCTIVE", False),
        upgrade_proposal_enabled=_env_bool("EVOLUTION_UPGRADE_PROPOSAL_ENABLED", True),
        upgrade_auto_validate=_env_bool("EVOLUTION_UPGRADE_AUTO_VALIDATE", True),
        upgrade_human_approval_required=_env_bool("EVOLUTION_UPGRADE_HUMAN_APPROVAL_REQUIRED", False),
        upgrade_apply_enabled=_env_bool("EVOLUTION_UPGRADE_APPLY_ENABLED", True),
        upgrade_deploy_enabled=_env_bool("EVOLUTION_UPGRADE_DEPLOY_ENABLED", True),
        upgrade_max_files_per_proposal=max(1, min(int(os.getenv("EVOLUTION_UPGRADE_MAX_FILES_PER_PROPOSAL", "3") or "3"), 3)),
    )
