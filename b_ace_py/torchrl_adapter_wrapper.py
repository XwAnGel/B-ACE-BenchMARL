import torch
import numpy as np
from tensordict import TensorDict
from torchrl.envs import EnvBase
from torchrl.data import Composite, Unbounded, DiscreteTensorSpec

class TorchRLAdapterWrapper(EnvBase):
    """
    TorchRL 适配器：将底层的 Agent-Level 数据映射为 BenchMARL 所需的 Group-Level 数据。
    
    修正版 v6: 
    1. 包含 v4/v5 的所有修复。
    2. [NEW] 在 Root 层级添加 "state"，满足 BenchMARL Task 定义的全局状态需求。
    """
    def __init__(self, env, device="cpu", batch_size=None):
        if batch_size is None:
            batch_size = torch.Size([])
        super().__init__(device=device, batch_size=batch_size)
        
        self.env = env
        self.possible_agents = self.env.possible_agents
        self.observation_spaces_pz = self.env.observation_spaces_pz
        self.action_spaces_pz = self.env.action_spaces_pz
        
        # 定义分组映射 (Agent -> Group)
        self.group_mapping = {"blue": [a for a in self.possible_agents]} 
        self._group_map = self.group_mapping

        # 构建 Specs
        self._make_specs(device)

    def _make_specs(self, device):
        # 计算全局状态维度 (所有 agent obs 拼接)
        # 假设所有 agent obs 维度相同
        first_agent = self.possible_agents[0]
        base_obs_shape = self.observation_spaces_pz[first_agent].shape
        global_state_dim = base_obs_shape[0] * len(self.possible_agents)

        # 1. Observation Spec (包含 Root State 和 Group Observation)
        obs_specs = {}
        
        # [NEW] 添加 Root Level State Spec
        # 这是为了匹配 b_ace.py 中定义的 state_spec
        obs_specs["state"] = Unbounded(shape=(global_state_dim,), dtype=torch.float32, device=device)

        for group, agents in self.group_mapping.items():
            if not agents: continue
            
            n_agents = len(agents)
            group_shape_obs = (n_agents, *base_obs_shape)
            
            # Spec for Group
            obs_specs[group] = Composite(
                {
                    "observation": Unbounded(shape=group_shape_obs, dtype=torch.float32, device=device),
                    # Group 内部也保留一份 State (供 Agent-Level Critic 使用)
                    "state": Unbounded(shape=(n_agents, global_state_dim), dtype=torch.float32, device=device)
                },
                shape=self.batch_size + torch.Size([n_agents]) 
            )
        self.observation_spec = Composite(obs_specs, shape=self.batch_size)

        # 2. Action Spec
        act_specs = {}
        for group, agents in self.group_mapping.items():
            if not agents: continue
            first_agent = agents[0]
            base_shape = self.action_spaces_pz[first_agent].shape
            n_agents = len(agents)
            
            group_shape = (n_agents, *base_shape)
            
            act_specs[group] = Composite(
                {"action": Unbounded(shape=group_shape, dtype=torch.float32, device=device)},
                shape=self.batch_size + torch.Size([n_agents])
            )
        self.action_spec = Composite(act_specs, shape=self.batch_size)
        
        # 3. Reward Spec
        reward_specs = {}
        for group, agents in self.group_mapping.items():
            if not agents: continue
            n_agents = len(agents)
            
            reward_specs[group] = Composite(
                {"reward": Unbounded(shape=(n_agents, 1), dtype=torch.float32, device=device)},
                shape=self.batch_size + torch.Size([n_agents])
            )
        self.reward_spec = Composite(reward_specs, shape=self.batch_size)

        # 4. Done Spec
        done_specs = {}
        for group, agents in self.group_mapping.items():
            if not agents: continue
            n_agents = len(agents)
            
            done_specs[group] = Composite(
                {
                    "done": DiscreteTensorSpec(2, shape=(n_agents, 1), dtype=torch.bool, device=device),
                    "terminated": DiscreteTensorSpec(2, shape=(n_agents, 1), dtype=torch.bool, device=device),
                    "truncated": DiscreteTensorSpec(2, shape=(n_agents, 1), dtype=torch.bool, device=device),
                },
                shape=self.batch_size + torch.Size([n_agents])
            )
        
        # 全局 Done
        done_specs["done"] = DiscreteTensorSpec(2, shape=(1,), dtype=torch.bool, device=device)
        done_specs["terminated"] = DiscreteTensorSpec(2, shape=(1,), dtype=torch.bool, device=device)
        done_specs["truncated"] = DiscreteTensorSpec(2, shape=(1,), dtype=torch.bool, device=device)
        
        self.done_spec = Composite(done_specs, shape=self.batch_size)

    @property
    def group_map(self):
        return self._group_map
    
    @property
    def reset_keys(self):
        return ["_reset"]

    def _convert_obs_to_numpy(self, obs_dict):
        for agent, data in obs_dict.items():
            if "obs" in data and not isinstance(data["obs"], np.ndarray):
                try:
                    data["obs"] = np.array(data["obs"], dtype=np.float32)
                except Exception:
                    obs_shape = self.observation_spaces_pz[agent].shape
                    data["obs"] = np.zeros(obs_shape, dtype=np.float32)
        return obs_dict

    def _construct_global_state(self, raw_observations):
        """
        辅助函数：将所有 Agent 的 Observation 拼接成一个 Global State 向量。
        """
        state_parts = []
        for agent in self.possible_agents:
            if agent in raw_observations:
                obs = raw_observations[agent]["obs"]
            else:
                obs = np.zeros(self.observation_spaces_pz[agent].shape, dtype=np.float32)
            state_parts.append(obs.flatten())
        
        if not state_parts:
            return np.array([], dtype=np.float32)
            
        global_state = np.concatenate(state_parts)
        return global_state

    def _reset(self, tensordict=None, **kwargs):
        raw_observations, raw_info = self.env.reset(**kwargs)
        raw_observations = self._convert_obs_to_numpy(raw_observations)
        
        # 构建 Global State
        global_state_np = self._construct_global_state(raw_observations)
        global_state_tensor = torch.as_tensor(global_state_np, dtype=torch.float32, device=self.device)
        
        out_dict = TensorDict({}, batch_size=self.batch_size, device=self.device)
        
        # [NEW] 将 State 放入 Root
        out_dict["state"] = global_state_tensor

        for group, agents in self.group_mapping.items():
            if not agents: continue
            
            group_obs_list = []
            for agent in agents:
                if agent in raw_observations:
                    group_obs_list.append(raw_observations[agent]["obs"])
                else:
                    shape = self.observation_spaces_pz[agent].shape
                    group_obs_list.append(np.zeros(shape, dtype=np.float32))
            
            group_obs_np = np.stack(group_obs_list, axis=0)
            n_agents = len(agents)
            
            # 将 Global State 扩展到 (n_agents, state_dim) 放入组内
            group_state_tensor = global_state_tensor.unsqueeze(0).expand(n_agents, -1)
            
            out_dict[group] = TensorDict(
                {
                    "observation": torch.as_tensor(group_obs_np, dtype=torch.float32, device=self.device),
                    "state": group_state_tensor, 
                    "done": torch.zeros((n_agents, 1), dtype=torch.bool, device=self.device),
                    "terminated": torch.zeros((n_agents, 1), dtype=torch.bool, device=self.device),
                    "truncated": torch.zeros((n_agents, 1), dtype=torch.bool, device=self.device)
                },
                batch_size=self.batch_size + torch.Size([n_agents]),
                device=self.device
            )

        out_dict["done"] = torch.zeros((1,), dtype=torch.bool, device=self.device)
        out_dict["terminated"] = torch.zeros((1,), dtype=torch.bool, device=self.device)
        out_dict["truncated"] = torch.zeros((1,), dtype=torch.bool, device=self.device)

        return out_dict

    def _step(self, tensordict):
        actions_dict = {}
        for group, agents in self.group_mapping.items():
            if group in tensordict.keys():
                group_action = tensordict[group].get("action")
                if group_action is not None:
                    group_action_np = group_action.detach().cpu().numpy()
                    
                    if not np.all(np.isfinite(group_action_np)):
                        group_action_np = np.nan_to_num(group_action_np, nan=0.0, posinf=1.0, neginf=-1.0)

                    for i, agent in enumerate(agents):
                        if i < len(group_action_np):
                             actions_dict[agent] = group_action_np[i].tolist()
        
        for agent in self.possible_agents:
            if agent not in actions_dict:
                 actions_dict[agent] = [0.0] * self.action_spaces_pz[agent].shape[0]

        obs, rewards, terms, truncs, infos = self.env.step(actions_dict)
        obs = self._convert_obs_to_numpy(obs)

        global_state_np = self._construct_global_state(obs)
        global_state_tensor = torch.as_tensor(global_state_np, dtype=torch.float32, device=self.device)

        out_dict = TensorDict({}, batch_size=self.batch_size, device=self.device)
        
        # [NEW] 将 State 放入 Root
        out_dict["state"] = global_state_tensor

        is_reward_scalar = np.isscalar(rewards) or isinstance(rewards, (int, float))
        is_term_scalar = np.isscalar(terms) or isinstance(terms, (bool, np.bool_))
        is_trunc_scalar = np.isscalar(truncs) or isinstance(truncs, (bool, np.bool_))
        
        global_term = False
        global_trunc = False

        for group, agents in self.group_mapping.items():
            if not agents: continue
            
            group_obs_list = []
            group_rew_list = []
            group_term_list = []
            group_trunc_list = []
            group_done_list = []

            for agent in agents:
                if agent in obs:
                    group_obs_list.append(obs[agent]["obs"])
                else:
                    group_obs_list.append(np.zeros(self.observation_spaces_pz[agent].shape, dtype=np.float32))
                
                if is_reward_scalar: r = float(rewards)
                else: r = float(rewards.get(agent, 0.0))
                group_rew_list.append([r])

                if is_term_scalar: t = bool(terms)
                else: t = bool(terms.get(agent, False))
                
                if is_trunc_scalar: tr = bool(truncs)
                else: tr = bool(truncs.get(agent, False))
                
                group_term_list.append([t])
                group_trunc_list.append([tr])
                group_done_list.append([t or tr])

                if t: global_term = True
                if tr: global_trunc = True

            n_agents = len(agents)
            group_state_tensor = global_state_tensor.unsqueeze(0).expand(n_agents, -1)

            out_dict[group] = TensorDict(
                {
                    "observation": torch.as_tensor(np.stack(group_obs_list), dtype=torch.float32, device=self.device),
                    "state": group_state_tensor,
                    "reward": torch.as_tensor(np.stack(group_rew_list), dtype=torch.float32, device=self.device),
                    "done": torch.as_tensor(np.stack(group_done_list), dtype=torch.bool, device=self.device),
                    "terminated": torch.as_tensor(np.stack(group_term_list), dtype=torch.bool, device=self.device),
                    "truncated": torch.as_tensor(np.stack(group_trunc_list), dtype=torch.bool, device=self.device),
                },
                batch_size=self.batch_size + torch.Size([n_agents]),
                device=self.device
            )

        out_dict["done"] = torch.as_tensor([global_term or global_trunc], dtype=torch.bool, device=self.device)
        out_dict["terminated"] = torch.as_tensor([global_term], dtype=torch.bool, device=self.device)
        out_dict["truncated"] = torch.as_tensor([global_trunc], dtype=torch.bool, device=self.device)
        
        return out_dict

    def _set_seed(self, seed: int | None):
        return seed

    def __getattr__(self, name):
        return getattr(self.env, name)