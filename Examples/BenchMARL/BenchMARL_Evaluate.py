import argparse
import sys
import os
import torch
import numpy as np
import time
from pathlib import Path
from benchmarl.hydra_config import reload_experiment_from_file

# --- 1. 设置路径 (自动处理 Windows 路径) ---
current_file = Path(os.path.realpath(__file__))
# 假设脚本在 Examples/BenchMARL/ 下，向上 2 级是根目录
project_root = current_file.parent.parent.parent 
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def evaluate_model(checkpoint_path: str):
    # 处理 Windows 路径可能存在的引号问题
    checkpoint_path = checkpoint_path.strip('"').strip("'")
    print(f"Loading experiment from: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        print(f"❌ Error: File not found: {checkpoint_path}")
        return

    # --- 2. 加载实验 ---
    try:
        experiment = reload_experiment_from_file(checkpoint_path)
    except Exception as e:
        print(f"\n❌ Error loading checkpoint. If this model was trained on Linux and moved to Windows, paths might differ.")
        print(f"Error details: {e}")
        return

    # --- 3. 配置覆盖 (Windows 豪华可视版) ---
    experiment.config.on_policy_n_envs_per_worker = 1
    experiment.config.off_policy_n_envs_per_worker = 1
    experiment.config.evaluation_deterministic_actions = True 
    
    # 环境配置
    env_config = experiment.task.config["EnvConfig"]
    
    # 【Windows 专属配置】
    env_config["renderize"] = 0       # ✅ 开启画面！
    env_config["speed_up"] = 50000        # ✅ 1倍速 (真实速度)，方便肉眼观察
    env_config["port"] = 22000        # 独立端口
    
    print("\n--- Starting Windows Evaluation (Visual Mode) ---")
    print(f"Renderize: {env_config['renderize']}")
    print(f"Speed Up:  {env_config['speed_up']}")

    # --- 4. 创建环境 (CPU) ---
    # 依然保持 1v1 (num_envs=1)，防止形状不匹配
    env_func = experiment.task.get_env_fun(
        num_envs=1, 
        continuous_actions=True, 
        seed=0, 
        device="cpu" 
    )
    env = env_func()
    
    # --- 5. 准备策略网络 (CPU) ---
    policy = experiment.policy
    policy = policy.cpu() # 修复设备冲突
    policy.eval()

    total_episodes = 3 # 看 3 局就够了
    
    try:
        for i in range(total_episodes):
            print(f"\n>>> Episode {i+1}/{total_episodes} Starting... (Please check the Godot Window)")
            tensordict = env.reset()
            
            episode_reward = 0
            step_count = 0
            
            while True:
                
                # 1. 推理
                with torch.no_grad():
                    tensordict = policy(tensordict)
                
                # 2. 步进
                tensordict = env.step(tensordict)
                
                # 3. 统计奖励
                reward = 0.0
                reward_tensor = tensordict.get(("next", "reward"), None)
                if reward_tensor is None:
                    # MAPPO 的奖励路径可能略有不同，这里做兼容
                    if ("next", "agents", "reward") in tensordict.keys(include_nested=True):
                        reward_tensor = tensordict.get(("next", "agents", "reward"))
                    elif ("next", "blue", "reward") in tensordict.keys(include_nested=True):
                        reward_tensor = tensordict.get(("next", "blue", "reward"))

                if reward_tensor is not None:
                    reward = reward_tensor.sum().item()
                
                episode_reward += reward
                step_count += 1
                
                # 4. 获取 Done
                done = tensordict.get(("next", "done"), torch.tensor(False)).any().item()

                # 5. 打印状态 (每 50 帧)
                if step_count % 50 == 0 or done:
                    print(f"Step {step_count:04d} | Reward: {reward:.4f} | Done: {done}")

                # 6. 结束检测
                if done:
                    print(f">>> Episode Finished! Total Reward: {episode_reward:.4f}")
                    break 

                # 下一帧
                tensordict = tensordict["next"]

    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")
    except Exception as e:
        print(f"\nRuntime Error: {e}")
        if "shape[-2]" in str(e):
             print("\n⚠️ 诊断: 发生了形状匹配错误！")
             print("这说明你的模型是 1v1 的，但环境试图运行 6v6 (或者反过来)。")
             print("MAPPO 对输入维度非常敏感。")
    finally:
        print("Closing Environment...")
        env.close()

if __name__ == "__main__":
    # 硬编码路径方便测试，或者继续用 argparse
    # 这里为了方便你直接复制运行，我保留 argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=r"E:\angel\B-ACE-main\Results\mappo_b_ace_mlp__4a1e1b59_25_12_15-20_53_27\checkpoints\checkpoint_360000.pt", help="Path to checkpoint")
    args = parser.parse_args()
    
    evaluate_model(args.checkpoint)