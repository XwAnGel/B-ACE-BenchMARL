import argparse
import sys
import os
import json
from pathlib import Path
from typing import List, Any

current_file = Path(os.path.realpath(__file__))

# 假设 B-ACE-main 是项目根目录。
# 如果 BenchMARL_Training.py 在 Examples/BenchMARL/ 下，
# 则 project_root 应该是 current_file 向上两级
project_root = current_file.parent.parent.parent # 假设 Examples/BenchMARL/BenchMARL_Training.py
# 如果 BenchMARL_Training.py 就在 Examples/ 下，则是向上两级
# 如果您确定 B-ACE-main 是根目录，且 b_ace_py 在其中，使用：
# project_root = Path("/home/angel/B-ACE-main") # <--- 推荐直接使用绝对路径作为测试

# 重新计算 project_root: 
# 如果文件是 /home/angel/B-ACE-main/Examples/BenchMARL/BenchMARL_Training.py
# 那么 B-ACE-main 就是向上三级
project_root = current_file.parent.parent.parent 


# 确保项目根目录 (B-ACE-main) 被添加到 Python 搜索路径
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    print(f"Added project root to sys.path: {project_root}")

# --- BenchMARL 和 B-ACE 导入 ---
import torch
from tensordict import TensorDictBase
from benchmarl.environments.godotrl import b_ace # 导入我们修正后的 B_ACE Task Class
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig
from benchmarl.algorithms import IppoConfig, IsacConfig, IqlConfig, IddpgConfig
from benchmarl.algorithms import QmixConfig, VdnConfig, MappoConfig, MaddpgConfig, MasacConfig
from benchmarl.experiment.callback import Callback
from b_ace_py.utils import load_b_ace_config 

# --- 辅助函数：类型转换和配置更新 ---

def convert_str_to_type(value: str) -> Any:
    """尝试将字符串转换为 int, float, bool 或保持为字符串。"""
    if value.lower() in ('true', 'false'):
        return value.lower() == 'true'
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value

def update_dict(config_dict: dict, key_path: str, value: str):
    """根据 'Key.Path.To.Value' 字符串更新配置字典，并尝试进行类型转换。"""
    keys = key_path.split('.')
    current_dict = config_dict
    
    converted_value = convert_str_to_type(value)

    for key in keys[:-1]:
        if key not in current_dict:
            raise KeyError(f"Error: Key '{key}' not found in the configuration dictionary at path '{key_path}'.")
        current_dict = current_dict[key]
    
    final_key = keys[-1]
    if final_key not in current_dict:
        # 如果键不存在，我们允许创建新键，但这在 B-ACE 预设配置中不常见
        pass
        
    current_dict[final_key] = converted_value
    print(f"Updated: {key_path} -> {converted_value} (Type: {type(converted_value).__name__})")

# --- 回调函数：保存最佳检查点 ---

class SaveBest(Callback):
    """自定义回调函数，用于在评估结束后保存具有最佳平均奖励的模型。"""
    def __init__(self, save_best_folder: str = "Best"):
        # 将初始最佳奖励设置为负无穷
        self.best_mean_reward = -float('inf') 
        self.save_best_folder = save_best_folder
            
    def on_evaluation_end(self, rollouts: List[TensorDictBase]):
        current_mean_return = self.experiment.mean_return
        
        if current_mean_return > self.best_mean_reward:
            print(f"New Best Mean Return Found: {current_mean_return:.4f} (Previous Best: {self.best_mean_reward:.4f})")
            self.best_mean_reward = current_mean_return
            
            # 暂时修改保存路径到特定的 Best 文件夹
            original_folder = self.experiment.folder_name
            self.experiment.folder_name = Path(original_folder) / self.save_best_folder
            
            # 使用内部方法保存检查点
            self.experiment._save_experiment(checkpoint_name="checkpoint_best.pt")
            
            # 恢复原始保存路径
            self.experiment.folder_name = original_folder

# --- 主运行逻辑 ---

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Run benchmarl experiments for B-ACE.')
    parser.add_argument(
        '--algorithm', 
        type=str, 
        default='mappo', 
        choices=['ippo', 'isac', 'iql', 'qmix', 'vdn', 'mappo', 'maddpg', 'iddpg', 'masac'], 
        help='Algorithm configuration to use.'
    )
    # 使用 action='store_true' 处理布尔值参数
    parser.add_argument(
        '--savebest', 
        action='store_true', 
        help='Set flag to save Checkpoint of best rewards.'
    )
    parser.add_argument(
        '--config', 
        nargs='*', 
        action='append', 
        help='Key-value pairs to update the b_ace_config (e.g., --config EnvConfig.renderize=1 AgentsConfig.blue_agents.num_agents=2).'
    )
    
    args = parser.parse_args()
    
    # --- 1. 实验配置 (ExperimentConfig) ---
    
    experiment_config = ExperimentConfig.get_from_yaml() 

    # 训练/采样设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training Device set to: {device}")
    experiment_config.sampling_device = device
    experiment_config.train_device = device    
    
    # 训练时长
    experiment_config.max_n_frames = int(1e7)
    experiment_config.max_n_iters = 500
    experiment_config.checkpoint_interval = 120000 
    
    # 策略/评估设置
    experiment_config.share_policy_params = True
    experiment_config.prefer_continuous_actions = True 
    experiment_config.evaluation_interval = 60000 
    experiment_config.evaluation_episodes = 10 
    experiment_config.evaluation_deterministic_actions = False 
    
    # 探索率设置
    experiment_config.exploration_eps_init = 0.70
    experiment_config.exploration_eps_end = 0.01 
    
    # On-Policy 配置 (如 IPPO, MAPPO)
    experiment_config.on_policy_collected_frames_per_batch = 12000 
    experiment_config.on_policy_n_minibatch_iters = 5 
    experiment_config.on_policy_minibatch_size = 2048 
    
    # Off-Policy 配置 (如 ISAC, MADDPG, QMIX)
    experiment_config.off_policy_collected_frames_per_batch = 12000
    experiment_config.off_policy_n_optimizer_steps = 32 
    experiment_config.off_policy_train_batch_size = 1024 
    experiment_config.off_policy_memory_size = 100_000 
    experiment_config.off_policy_init_random_frames = 0
    
    # 环境并行数
    experiment_config.off_policy_n_envs_per_worker = 4
    experiment_config.on_policy_n_envs_per_worker = 4
    
    experiment_config.lr = 0.000001
    experiment_config.save_folder = "Results" 
    
    # --- 2. 任务配置 (Task Config) ---

    # 加载基础配置
    # 假设 Default_B_ACE_config.json 位于 b_ace_py 目录下
    config_path = str(project_root / "b_ace_py/Default_B_ACE_config.json")
    b_ace_config = load_b_ace_config(config_path) 
    
    # 默认覆盖
    b_ace_config["EnvConfig"]["env_path"] = "bin/B_ACE_v0.1.exe" 
    b_ace_config["EnvConfig"]["renderize"] = 0 

    # 1. 增大“持续锁定”的奖励 (原 0.001 -> 0.01 或 0.05)
    # 只要敌人处于攻击锥内，就持续给分。这能让 AI 快速学会“机头对准敌人”。
    b_ace_config["EnvConfig"]["RewardsConfig"]["keep_track_factor"] = 0.02 
    
    # 2. 稍微增大“任务/存活”奖励 (原 0.001 -> 0.005)
    # 鼓励它们活久一点，不要开局就撞地。
    b_ace_config["EnvConfig"]["RewardsConfig"]["mission_factor"] = 0.005

    # 3. 减小“导弹脱靶”惩罚 (原 -0.5 -> -0.1)
    # 6v6 场面混乱，初期很难命中。如果惩罚太重，AI 会学会“永远不开火”来避免扣分。
    b_ace_config["EnvConfig"]["RewardsConfig"]["missile_miss_factor"] = -0.1
    
    # 4. 保持高额击杀奖励 (引导最终目标)
    # 这个保持 3.0 或增加到 5.0 都可以，确保击杀是最赚的。
    b_ace_config["EnvConfig"]["RewardsConfig"]["hit_enemy_factor"] = 4.0
    
    # 从命令行更新配置
    if args.config:
        for config_arg_list in args.config:
            for param in config_arg_list:
                key_value = param.split('=', 1) 
                if len(key_value) != 2:
                    print(f"Error: Invalid configuration argument format: {param}. Expected key=value.")
                    sys.exit(1)
                key_path, value = key_value
                try:
                    update_dict(b_ace_config, key_path, value)
                except (KeyError, ValueError) as e:
                    print(f"Failed to update configuration for {key_path}: {e}")
                    sys.exit(1)

    print("\n--- Updated B-ACE Configuration ---")
    print(json.dumps(b_ace_config, indent=4))
    print("-----------------------------------")

    # 初始化任务
    task = b_ace.B_ACE(config=b_ace_config)
    
    # --- 3. 算法、模型配置 ---
    
    # Set the algorithm configuration based on the provided argument
    alg = args.algorithm.lower()
    if alg == 'ippo':
        algorithm_config = IppoConfig.get_from_yaml()
    elif alg == 'isac':
        algorithm_config = IsacConfig.get_from_yaml()
    elif alg == 'iql':
        algorithm_config = IqlConfig.get_from_yaml()
    elif alg == 'qmix':
        algorithm_config = QmixConfig.get_from_yaml()
    elif alg == 'vdn':
        algorithm_config = VdnConfig.get_from_yaml()
    elif alg == 'mappo':
        algorithm_config = MappoConfig.get_from_yaml()
    elif alg == 'maddpg':
        algorithm_config = MaddpgConfig.get_from_yaml()
    elif alg == 'iddpg':
        algorithm_config = IddpgConfig.get_from_yaml()
    elif alg == 'masac':
        algorithm_config = MasacConfig.get_from_yaml()
    else: 
        raise ValueError(f"Unknown algorithm: {args.algorithm}")
    
    print(f"Configuring Gradient Clipping for {alg}...")
    
    # 对于 PPO/MAPPO 类算法 (On-Policy)
    if hasattr(algorithm_config, "max_grad_norm"):
        algorithm_config.max_grad_norm = 1.0 # 允许的最大梯度范数，通常 1.0 到 10.0
        print(" -> Set max_grad_norm = 1.0")
        
    if hasattr(algorithm_config, "clip_epsilon"):
        algorithm_config.clip_epsilon = 0.2 # PPO 的截断范围，防止策略更新太猛
        print(" -> Set clip_epsilon = 0.2")

    # 对于 QMIX/DDPG 类算法 (Off-Policy) 有些可能叫 clip_grad_val
    if hasattr(algorithm_config, "clip_grad_val"):
        algorithm_config.clip_grad_val = 1.0
        print(" -> Set clip_grad_val = 1.0")
    
    if hasattr(algorithm_config, "entropy_coef"):
        algorithm_config.entropy_coef = 0.0  # <--- 新增这行！强制为 0
        print(" -> Set entropy_coef = 0.0 (Prevent Variance Explosion)")

    # 【额外建议】：如果用 MAPPO，开启 Value Function 的标准化有助于收敛
    if hasattr(algorithm_config, "standardize_advantages"):
        algorithm_config.standardize_advantages = True
        print(" -> Enabled advantage standardization")
        
    # 模型配置
    model_config = MlpConfig.get_from_yaml()
    critic_model_config = MlpConfig.get_from_yaml()

    model_config.layer_normalization = True     # <--- 加上这行
    critic_model_config.layer_normalization = True # <--- 加上这行

    model_config.layers = [256, 256, 256]
    critic_model_config.layers = [512, 256, 256]

    # --- 4. 实验运行 ---
    
    # 配置回调函数
    callbacks_list = []
    if args.savebest:
        callbacks_list.append(SaveBest())
        print("Note: SaveBest callback enabled.")
    
    # 运行多个种子（seed=3 和 seed=4）
    for i in range(3, 5): 

        experiment = Experiment(
            task=task,
            algorithm_config=algorithm_config,
            model_config=model_config,
            critic_model_config=critic_model_config,
            seed=i, 
            config=experiment_config,
            callbacks=callbacks_list,
        )
            
        print(f"\n--- Starting Experiment for Seed {i} with {alg.upper()} ---")
        experiment.run()