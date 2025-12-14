import sys
from pathlib import Path
from typing import Callable, Dict, Any, List
import yaml
import torch
from gymnasium import spaces
from benchmarl.environments import PettingZooTask
from torchrl.envs import EnvBase, RewardSum, Transform
# from benchmarl.environments.common import get_reward_sum_transform 
import sys
from pathlib import Path
from torchrl.data import TensorSpec

# 直接将项目根目录（~/B-ACE-main）添加到 Python 路径
project_root = Path.home() / "B-ACE-main"
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from b_ace_py.B_ACE_GodotPettingZooWrapper import B_ACE_GodotPettingZooWrapper
from b_ace_py.utils import load_b_ace_config

class B_ACE:
    """B-ACE环境封装类，适配benchmarl框架的PettingZoo接口"""
    b_ace = None  # 类变量存储环境实例

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.env_config = self.config["EnvConfig"]  # 从配置中提取环境配置
        self.agents_config = self.config["AgentsConfig"]  # 提取智能体配置
        self.name = self.env_config["task"]  # 任务名称（如"b_ace_v1"）
        self._env = None  # 缓存环境实例

    @classmethod
    def get_from_yaml(cls, yaml_path: str = None) -> PettingZooTask:
        """从YAML加载配置或使用默认配置，创建benchmarl兼容任务"""
        # 1. 加载默认配置（若未提供YAML）
        if yaml_path is None:
            default_config_path = Path(__file__).parent / ".." / "b_ace_py" / "Default_B_ACE_config.json"
            config = load_b_ace_config(str(default_config_path))
        else:
            # 2. 从YAML加载用户配置
            with open(Path(yaml_path), "r") as f:
                config = yaml.safe_load(f)
        
        # 3. 创建B_ACE实例并包装为PettingZooTask
        cls.b_ace = cls(config)._create_task()
        return cls.b_ace

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> PettingZooTask:
        """从字典配置创建环境（适配您提供的b_ace_config）"""
        cls.b_ace = cls(config_dict)._create_task()
        return cls.b_ace

    def _create_task(self) -> PettingZooTask:
        """创建PettingZoo环境并转换为benchmarl任务"""
        env_fun = self.get_env_fun()
        env = env_fun()  # 初始化原始环境（B_ACE_GodotPettingZooWrapper或B_ACE_TaskEnv）
        
        # 创建PettingZooTask时，确保原始环境可被访问
        task = PettingZooTask(
            env=env,
            env_name=self.env_name(),
            action_space=self.get_action_spaces(),
            observation_space=self.get_observation_spaces(),
            agents=env.possible_agents,
            num_agents=len(env.possible_agents)
        )
        
        # 为转换后的环境添加base_env属性，指向原始环境
        task.base_env = env  # 关键：保留原始环境引用
        return task
    @staticmethod
    # 在 B_ACE 任务类的 observation_spec 方法中
    # def observation_spec(self, env):
    #     # 逐层剥离包装器（处理 TransformedEnv 和 PettingZooWrapper 等）
    #     base_env = env
    #     # 循环解析：先处理 TransformedEnv 的 base_env，再处理其他包装器的 env
    #     while hasattr(base_env, 'base_env') or hasattr(base_env, 'env'):
    #         if hasattr(base_env, 'base_env'):
    #             base_env = base_env.base_env  # 剥离 TransformedEnv
    #         elif hasattr(base_env, 'env'):
    #             base_env = base_env.env  # 剥离 PettingZooWrapper 等
    #     # 此时 base_env 应为原始的 B_ACE_GodotPettingZooWrapper 或 B_ACE_TaskEnv
        
    #     # 验证原始环境类型（可选，用于调试）
    #     if not hasattr(base_env, 'possible_agents'):
    #         raise AttributeError("原始环境缺少 possible_agents 属性")
        
    #     # 构建观测空间规格
    #     observation_spec = {
    #         agent: {
    #             "observation": TensorSpec(
    #                 shape=base_env.observation_space['obs'].shape,  # 从原始环境获取形状
    #                 dtype=torch.float32,
    #                 device=self.device,
    #             )
    #         } for agent in base_env.possible_agents
    #     }
    #     return observation_spec
    def observation_spec(self, env: EnvBase) -> Composite:
        observation_spec = env.observation_spec.clone()
        for group_key in list(observation_spec.keys()):
            if group_key not in self.group_map(env).keys():
                del observation_spec[group_key]
        return observation_spec

    @staticmethod
    def info_spec(env):
        """返回每个智能体的信息空间规格（如 agent_id）"""
        info_spec = {}
        for agent in env.possible_agents:
            info_spec[agent] = {
                "agent_id": {
                    "shape": (),  # 标量
                    "dtype": torch.int64  # 用整数标识智能体
                }
            }
        return info_spec

    @staticmethod
    def state_spec(env):
        """返回全局状态规格（若环境无全局状态，返回空字典）"""
        return {}  # B-ACE 环境通常无全局状态，可留空

    @staticmethod
    def action_mask_spec(env):
        """返回动作掩码规格（若使用动作掩码）"""
        mask_spec = {}
        for agent in env.possible_agents:
            # 动作掩码形状与动作空间维度一致（离散动作适用）
            action_space = env.action_space(agent)
            if isinstance(action_space, spaces.Discrete):
                mask_shape = (action_space.n,)
            else:  # 连续动作通常无掩码
                mask_shape = ()
            mask_spec[agent] = {
                "mask": {
                    "shape": mask_shape,
                    "dtype": torch.bool  # 掩码为布尔型
                }
            }
        return mask_spec

    @staticmethod
    def action_spec(env):
        """返回每个智能体的动作空间规格"""
        action_spec = {}
        for agent in env.possible_agents:
            action_space = env.action_space(agent)
            # 连续动作（Low_Level_Continuous）
            if isinstance(action_space, spaces.Box):
                action_spec[agent] = {
                    "input": {
                        "shape": action_space.shape,
                        "dtype": torch.float32
                    }
                }
            # 离散动作（Low_Level_Discrete）
            elif isinstance(action_space, spaces.Discrete):
                action_spec[agent] = {
                    "input": {
                        "shape": (),  # 离散动作是标量
                        "dtype": torch.int64
                    }
                }
        return action_spec

    @staticmethod
    def group_map(env):
        """返回智能体分组（默认所有智能体属于同一组）"""
        # 若有红蓝分组，可按 agent_id 区分（如 "blue_0", "red_0"）
        return {"all_agents": env.possible_agents}

    @staticmethod
    def max_steps(env):
        """返回每个episode的最大步数（从环境配置中获取）"""
        return env.env_config.get("max_cycles", 36000)  # 对应配置中的 max_cycles
    
    # 在 B_ACE 类中更新 get_env_fun 方法，新增 device 参数支持
    def get_env_fun(self, num_envs: int = 1, continuous_actions: bool = True, seed: int = None, device: str = "cpu") -> Callable[[], B_ACE_GodotPettingZooWrapper]:
        """返回环境创建函数（适配框架的设备、种子、并行环境和动作空间参数）"""
        def env_fun() -> B_ACE_GodotPettingZooWrapper:
            if self._env is None:
                # 复制基础配置并更新参数
                env_config = self.config.copy()
                env_config["EnvConfig"]["parallel_envs"] = num_envs
                # 根据 continuous_actions 调整动作类型配置
                if continuous_actions:
                    env_config["EnvConfig"]["action_type"] = "Low_Level_Continuous"
                else:
                    env_config["EnvConfig"]["action_type"] = "Low_Level_Discrete"
                # 应用种子参数（如果框架传入）
                if seed is not None:
                    env_config["EnvConfig"]["seed"] = seed
                # 初始化环境时传入 device 参数
                self._env = B_ACE_GodotPettingZooWrapper(
                    device=device,  # 使用框架传入的设备参数
                    **env_config
                )
            return self._env
        return env_fun
    def get_reward_sum_transform(self, env: EnvBase) -> Transform:
        """
        Returns the RewardSum transform for the environment

        Args:
            env (EnvBase): An environment created via self.get_env_fun
        """
        if "_reset" in env.reset_keys:
            reset_keys = ["_reset"] * len(self.group_map(env).keys())
        else:
            reset_keys = env.reset_keys
        return RewardSum(reset_keys=reset_keys)

    def get_observation_spaces(self) -> Dict[str, spaces.Space]:
        """返回每个智能体的观测空间"""
        env = self.get_env_fun()()
        return {agent: env.observation_spaces[agent] for agent in env.possible_agents}

    def get_action_spaces(self) -> Dict[str, spaces.Space]:
        """返回每个智能体的动作空间"""
        env = self.get_env_fun()()
        return {agent: env.action_spaces[agent] for agent in env.possible_agents}

    def supports_continuous_actions(self) -> bool:
        """判断是否支持连续动作（基于配置中的action_type）"""
        return self.env_config["action_type"] == "Low_Level_Continuous"

    def supports_discrete_actions(self) -> bool:
        """判断是否支持离散动作"""
        return self.env_config["action_type"] == "Low_Level_Discrete"

    def env_name(self) -> str:
        """返回环境名称"""
        return self.name

    def num_agents(self) -> int:
        """返回蓝方智能体数量（可根据需求扩展为总数量）"""
        return self.agents_config["blue_agents"]["num_agents"]
    def get_env_transforms(self, env):
    # 若无需转换，返回空列表
        return []
    # 或根据需求返回框架预设的转换（如归一化）

    def close(self) -> None:
        """关闭环境释放资源"""
        if self._env is not None:
            self._env.close()
            self._env = None