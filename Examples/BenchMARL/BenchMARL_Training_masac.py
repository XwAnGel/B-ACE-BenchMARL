import argparse
import sys
import os
import json
import math  # 引入 math 库用于向下取整/计算倍数
from pathlib import Path
from typing import List, Any

# --- 1. 基础路径设置 ---
current_file = Path(os.path.realpath(__file__))
project_root = current_file.parent.parent.parent 
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
from tensordict import TensorDictBase
from benchmarl.environments.godotrl import b_ace 
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig
from benchmarl.algorithms import IppoConfig, IsacConfig, IqlConfig, IddpgConfig
from benchmarl.algorithms import QmixConfig, VdnConfig, MappoConfig, MaddpgConfig, MasacConfig
from benchmarl.experiment.callback import Callback
from b_ace_py.utils import load_b_ace_config 

# --- 辅助函数 ---
def convert_str_to_type(value: str) -> Any:
    if value.lower() in ('true', 'false'): return value.lower() == 'true'
    try: return int(value)
    except ValueError:
        try: return float(value)
        except ValueError: return value

def update_dict(config_dict: dict, key_path: str, value: str):
    keys = key_path.split('.')
    current_dict = config_dict
    converted_value = convert_str_to_type(value)
    for key in keys[:-1]:
        current_dict = current_dict[key]
    current_dict[keys[-1]] = converted_value

# --- 回调函数 ---
class SaveBest(Callback):
    def __init__(self, save_best_folder: str = "Best"):
        self.best_mean_reward = -float('inf') 
        self.save_best_folder = save_best_folder
    def on_evaluation_end(self, rollouts: List[TensorDictBase]):
        current_mean_return = self.experiment.mean_return
        if current_mean_return > self.best_mean_reward:
            print(f"New Best Mean Return: {current_mean_return:.4f}")
            self.best_mean_reward = current_mean_return
            original_folder = self.experiment.folder_name
            self.experiment.folder_name = Path(original_folder) / self.save_best_folder
            self.experiment._save_experiment(checkpoint_name="checkpoint_best.pt")
            self.experiment.folder_name = original_folder

# ==============================================================================
#  核心逻辑：算法参数分发器 (已修复对齐问题)
# ==============================================================================

def get_algorithm_config(alg_name: str):
    """根据名称返回算法配置类"""
    alg_map = {
        'ippo': IppoConfig, 'mappo': MappoConfig,  # On-Policy
        'masac': MasacConfig, 'isac': IsacConfig, 'maddpg': MaddpgConfig, 'iddpg': IddpgConfig, # Off-Policy (AC)
        'qmix': QmixConfig, 'vdn': VdnConfig, 'iql': IqlConfig # Off-Policy (Value-Based)
    }
    if alg_name not in alg_map:
        raise ValueError(f"Unknown algorithm: {alg_name}")
    return alg_map[alg_name].get_from_yaml()

def configure_experiment_hyperparameters(exp_config: ExperimentConfig, alg_name: str, device: str):
    """
    根据算法是 On-Policy 还是 Off-Policy，自动调整实验超参数，并确保 Evaluation Interval 对齐。
    """
    # 1. 通用基础设置
    exp_config.max_n_frames = int(1e7)
    exp_config.max_n_iters = 2000 
    exp_config.evaluation_episodes = 10
    exp_config.share_policy_params = True
    exp_config.prefer_continuous_actions = True
    
    # 设备
    exp_config.sampling_device = device
    exp_config.train_device = device

    # 探索策略 (Exploration)
    exp_config.exploration_eps_init = 0.70
    exp_config.exploration_eps_end = 0.01
    
    # 2. 区分 On-Policy vs Off-Policy 并设置 collected_frames
    on_policy_algs = ['ippo', 'mappo']
    
    current_frames_per_batch = 0 # 用于后续对齐计算
    
    if alg_name in on_policy_algs:
        print(f"--- Configuring for ON-POLICY algorithm ({alg_name}) ---")
        current_frames_per_batch = 12_000
        
        exp_config.on_policy_collected_frames_per_batch = current_frames_per_batch
        exp_config.on_policy_n_minibatch_iters = 10      
        exp_config.on_policy_minibatch_size = 2048       
        exp_config.on_policy_n_envs_per_worker = 10     
        
    else:
        print(f"--- Configuring for OFF-POLICY algorithm ({alg_name}) ---")
        current_frames_per_batch = 6_000
        
        exp_config.off_policy_collected_frames_per_batch = current_frames_per_batch
        exp_config.off_policy_n_optimizer_steps = 100    
        exp_config.off_policy_train_batch_size = 256     
        exp_config.off_policy_memory_size = 1_000_000    
        exp_config.off_policy_init_random_frames = 10_000 
        exp_config.off_policy_n_envs_per_worker = 5

    # 3. 自动修复 Evaluation Interval 和 Checkpoint Interval 对齐问题
    # 目标：让 interval 成为 frames_per_batch 的整数倍
    
    target_eval_interval = 100_000
    target_ckpt_interval = 200_000
    
    # 计算倍数 (向上取整或四舍五入均可，这里用 round 找最近的倍数)
    eval_multiplier = max(1, round(target_eval_interval / current_frames_per_batch))
    ckpt_multiplier = max(1, round(target_ckpt_interval / current_frames_per_batch))
    
    aligned_eval_interval = eval_multiplier * current_frames_per_batch
    aligned_ckpt_interval = ckpt_multiplier * current_frames_per_batch
    
    exp_config.evaluation_interval = aligned_eval_interval
    exp_config.checkpoint_interval = aligned_ckpt_interval
    
    print(f"-> Auto-aligned Evaluation Interval: {aligned_eval_interval} (Multiple of {current_frames_per_batch})")
    print(f"-> Auto-aligned Checkpoint Interval: {aligned_ckpt_interval} (Multiple of {current_frames_per_batch})")

def configure_algorithm_hyperparameters(alg_config, alg_name: str):
    """
    针对具体算法微调内部参数
    """
    print(f"Configuring internal parameters for {alg_name}...")
    
    # 1. 梯度裁剪
    if hasattr(alg_config, "max_grad_norm"):
        alg_config.max_grad_norm = 10.0
    elif hasattr(alg_config, "clip_grad_val"): 
        alg_config.clip_grad_val = 10.0

    # 2. PPO/MAPPO 特有
    if alg_name in ['ippo', 'mappo']:
        alg_config.clip_epsilon = 0.2
        alg_config.entropy_coef = 0.01 
        if hasattr(alg_config, "standardize_advantages"):
            alg_config.standardize_advantages = True
            
    # 3. SAC/MASAC 特有
    if alg_name in ['isac', 'masac']:
        if hasattr(alg_config, "target_entropy"):
            alg_config.target_entropy = "auto" 
            print(" -> Set target_entropy = auto")
        
        if hasattr(alg_config, "polyak_tau"):
            alg_config.polyak_tau = 0.005 

    # 4. DDPG/MADDPG 特有
    if alg_name in ['iddpg', 'maddpg']:
        if hasattr(alg_config, "exploration_noise"):
            print(" -> Configured OU Noise for DDPG")

# ==============================================================================

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Run benchmarl experiments for B-ACE.')
    parser.add_argument('--algorithm', type=str, default='mappo', help='Algorithm to use (mappo, masac, etc.)')
    parser.add_argument('--savebest', action='store_true', help='Save best checkpoint.')
    parser.add_argument('--config', nargs='*', action='append', help='Update B-ACE config.')
    
    args = parser.parse_args()
    alg_name = args.algorithm.lower()
    
    # --- 1. 获取算法配置 ---
    try:
        algorithm_config = get_algorithm_config(alg_name)
        configure_algorithm_hyperparameters(algorithm_config, alg_name)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # --- 2. 实验配置 (ExperimentConfig) - 自动适配 On/Off Policy ---
    experiment_config = ExperimentConfig.get_from_yaml() 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 自动对齐 interval
    configure_experiment_hyperparameters(experiment_config, alg_name, device)
    
    experiment_config.lr = 5e-5 
    experiment_config.save_folder = "Results" 

    # --- 3. 任务配置 (B-ACE) ---
    config_path = str(project_root / "b_ace_py/Default_B_ACE_config.json")
    b_ace_config = load_b_ace_config(config_path) 
    
    # 默认 B-ACE 设置
    b_ace_config["EnvConfig"]["env_path"] = "bin/B_ACE_v0.1.exe" 
    b_ace_config["EnvConfig"]["renderize"] = 0 
    b_ace_config["EnvConfig"]["RewardsConfig"]["keep_track_factor"] = 0.02 
    b_ace_config["EnvConfig"]["RewardsConfig"]["mission_factor"] = 0.005
    b_ace_config["EnvConfig"]["RewardsConfig"]["missile_miss_factor"] = -0.1
    b_ace_config["EnvConfig"]["RewardsConfig"]["hit_enemy_factor"] = 4.0

    # 命令行覆盖配置
    if args.config:
        for config_arg_list in args.config:
            for param in config_arg_list:
                key_path, value = param.split('=', 1)
                update_dict(b_ace_config, key_path, value)

    print("\n--- B-ACE Configuration Ready ---")
    task = b_ace.B_ACE(config=b_ace_config)
    
    # --- 4. 网络模型配置 ---
    model_config = MlpConfig.get_from_yaml()
    critic_model_config = MlpConfig.get_from_yaml()

    model_config.layer_normalization = True 
    critic_model_config.layer_normalization = True
    model_config.layers = [256, 256, 256]
    critic_model_config.layers = [512, 256, 256]

    # --- 5. 运行实验 ---
    callbacks_list = [SaveBest()] if args.savebest else []

    # 单次运行
    for seed in range(3, 4):
        print(f"\n==================================================")
        print(f" STARTING TRAIN: {alg_name.upper()} | Seed: {seed}")
        print(f" Mode: {'ON-POLICY' if alg_name in ['ippo','mappo'] else 'OFF-POLICY'}")
        print(f" Auto-Aligned Eval Interval: {experiment_config.evaluation_interval}")
        print(f" Device: {device}")
        print(f"==================================================\n")
        
        experiment = Experiment(
            task=task,
            algorithm_config=algorithm_config,
            model_config=model_config,
            critic_model_config=critic_model_config,
            seed=seed, 
            config=experiment_config,
            callbacks=callbacks_list,
        )
        experiment.run()