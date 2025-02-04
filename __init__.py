"""
F1tenth locomotion environment.
"""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-F1tenth-Direct-v0",
    entry_point=f"{__name__}.f1tenth_env:F1tenthEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.f1tenth_env:F1tenthEnvCfg",
        # TODO: "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
