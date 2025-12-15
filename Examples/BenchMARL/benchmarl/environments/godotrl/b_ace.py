import copy
from typing import Callable, Dict, List, Optional
import torch
# 导入 Bounded 用于定义有界动作空间
from torchrl.data import Composite, Unbounded, Bounded 
from torchrl.envs import EnvBase, Transform
from torchrl.envs.transforms import DoubleToFloat
from benchmarl.environments.common import Task, TaskClass
from benchmarl.utils import DEVICE_TYPING

from b_ace_py.B_ACE_GodotPettingZooWrapper import B_ACE_GodotPettingZooWrapper
from b_ace_py.torchrl_adapter_wrapper import TorchRLAdapterWrapper

# --- 1. NaN 检测 ---
class DetectNaN(Transform):
    def _apply_transform(self, observation):
        if torch.isnan(observation).any() or torch.isinf(observation).any():
            raise ValueError(f"CRITICAL: Found NaN/Inf in observations! \nData: {observation}")
        return observation
        
    def _step(self, tensordict, next_tensordict):
        # 检查观测
        for key in ["observation"]: 
            obs = next_tensordict.get(("blue", key), None)
            if obs is not None:
                if torch.isnan(obs).any() or torch.isinf(obs).any():
                     raise ValueError(f"CRITICAL: Found NaN/Inf in NEW observation '{key}'")
        
        # 检查动作 (Action)
        if "action" in tensordict.keys(include_nested=True):
            action = tensordict.get(("blue", "action"), None)
            if action is not None:
                if torch.isnan(action).any() or torch.isinf(action).any():
                     print(f"Bad Action Tensor: {action}")
                     raise ValueError("CRITICAL: Actor network output NaN/Inf Action!")
                     
        return next_tensordict
    
    def transform_observation_spec(self, observation_spec):
        return observation_spec

# --- 2. 动作截断 (Action Clamp) ---
class ActionClamp(Transform):
    def __init__(self, min_val: float, max_val: float, in_keys: List[str]):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.in_keys = in_keys

    def forward(self, tensordict):
        for key in self.in_keys:
            if key in tensordict.keys(include_nested=True):
                val = tensordict.get(key)
                tensordict.set(key, torch.clamp(val, self.min_val, self.max_val))
        return tensordict

    def _step(self, tensordict, next_tensordict):
        return next_tensordict

# --- 主 TaskClass ---
class B_ACE(TaskClass):
    def __init__(self, config: Dict, name: str = "b_ace"):
        super().__init__(name=name, config=config)
    
    def group_map(self, env: EnvBase) -> Dict[str, List[str]]:
        if hasattr(env, "group_map"):
            return env.group_map
        n_agents = self.config["AgentsConfig"]["blue_agents"]["num_agents"]
        agents_list = [f"agent_{i}" for i in range(n_agents)]
        return {"blue": agents_list}
    
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
            raw_env = B_ACE_GodotPettingZooWrapper(device=device, **config)
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
        return [
            DoubleToFloat(),
            DetectNaN(),
            # 强制截断动作
            ActionClamp(min_val=-1.0, max_val=1.0, in_keys=[("blue", "action")]),
        ]

    def get_replay_buffer_transforms(self, env: EnvBase, group: str) -> List[Transform]:
        return []

    def state_spec(self, env: EnvBase) -> Optional[Composite]:
        obs_dim = 22 
        if hasattr(env, "observation_spec") and isinstance(env.observation_spec, Composite):
            if "blue" in env.observation_spec and "observation" in env.observation_spec["blue"]:
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
        return None

    # --- 【修正这里】 ---
    def action_spec(self, env: EnvBase) -> Composite:
        n_agents = self.config["AgentsConfig"]["blue_agents"]["num_agents"]
        action_dim = 4 
        
        # 1. 创建动作空间的具体定义 (Leaf Spec)
        action_leaf = Bounded(
            low=-1.0,
            high=1.0,
            shape=torch.Size([n_agents, action_dim]),
            dtype=torch.float32,
            device=env.device
        )

        # 2. 【关键】将其包裹在 'blue' 组名下
        # 这样 BenchMARL 才能通过 spec["blue"]["action"] 找到它
        return Composite(
            blue=Composite(
                action=action_leaf,
                shape=torch.Size([n_agents]) # 显式指定组的 batch shape (可选但推荐)
            )
        )

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