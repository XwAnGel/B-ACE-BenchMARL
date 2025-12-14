import copy
from typing import Callable, Dict, List, Optional
import torch
from torchrl.data import Composite, Unbounded
from torchrl.envs import DoubleToFloat, EnvBase, Transform
from benchmarl.environments.common import Task, TaskClass
from benchmarl.utils import DEVICE_TYPING

# 导入
from b_ace_py.B_ACE_GodotPettingZooWrapper import B_ACE_GodotPettingZooWrapper
from b_ace_py.torchrl_adapter_wrapper import TorchRLAdapterWrapper

class B_ACE(TaskClass):
    def __init__(self, config: Dict, name: str = "b_ace"):
        super().__init__(name=name, config=config)
    
    def group_map(self, env: EnvBase) -> Dict[str, List[str]]:
        if hasattr(env, "group_map"):
            return env.group_map
        return {"blue": ["agent_0"]} # 简化 Fallback

    def get_env_fun(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: Optional[int],
        device: DEVICE_TYPING,
    ) -> Callable[[], EnvBase]:
        config = copy.deepcopy(self.config)
        if seed is not None:
            config["EnvConfig"]["seed"] = seed
            
        def make_env():
            # 1. 实例化 Godot 环境
            raw_env = B_ACE_GodotPettingZooWrapper(device=device, **config)
            # 2. 使用新的适配器 (传递 device)
            wrapped_env = TorchRLAdapterWrapper(raw_env, device=device)
            return wrapped_env

        return make_env

    def supports_continuous_actions(self) -> bool:
        return True

    def supports_discrete_actions(self) -> bool:
        return False

    def has_render(self, env: EnvBase) -> bool:
        return self.config["EnvConfig"].get("renderize", 0) == 1

    def max_steps(self, env: EnvBase) -> int:
        return self.config["EnvConfig"].get("max_cycles", 36000)

    def get_env_transforms(self, env: EnvBase) -> List[Transform]:
        return [DoubleToFloat()]

    def get_replay_buffer_transforms(self, env: EnvBase, group: str) -> List[Transform]:
        return []

    def state_spec(self, env: EnvBase) -> Optional[Composite]:
        # 简单推断 State
        obs_dim = 22 
        if hasattr(env, "observation_spec") and "blue" in env.observation_spec:
             shape = env.observation_spec["blue"]["observation"].shape
             obs_dim = shape[-1]
        
        n_agents = self.config["AgentsConfig"]["blue_agents"]["num_agents"]
        return Composite(
            state=Unbounded(
                shape=torch.Size([n_agents * obs_dim]), 
                dtype=torch.float32, 
                device=env.device
            )
        )
    
    def observation_spec(self, env: EnvBase) -> Composite:
        return env.observation_spec

    def info_spec(self, env: EnvBase) -> Optional[Composite]:
        # 新版 EnvBase 实际上不强制要求 info_spec，但为了安全起见返回 None 或空
        return None

    def action_spec(self, env: EnvBase) -> Composite:
        return env.action_spec

    def action_mask_spec(self, env: EnvBase) -> Optional[Composite]:
        return None

    @staticmethod
    def env_name() -> str:
        return "b_ace"

    @staticmethod
    def render_callback(experiment, env: EnvBase, data):
        return None

class B_ACETask(Task):
    @staticmethod
    def associated_class():
        return B_ACE