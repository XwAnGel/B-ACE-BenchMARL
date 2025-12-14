# This is a simple example script demonstrating how to use the B_ACE_GodotRLPettingZooWrapper
# to interact with the B-ACE Godot environment from a Python script.
import numpy as np # 确保导入 numpy
from b_ace_py.utils import load_b_ace_config
from b_ace_py.B_ACE_GodotPettingZooWrapper import B_ACE_GodotPettingZooWrapper

# Define the environment configuration
B_ACE_config = load_b_ace_config('./b_ace_py/Default_B_ACE_config.json')

# Define desired environment configuration
env_config = { 
                "EnvConfig":{
                    "env_path": "./bin/B_ACE_v0.1.exe", # Path to the Godot executable
                    "renderize": 1,
                    "speed_up": 5000,
                }
            }
# Define desired agents configuration
agents_config = {
                    "AgentsConfig":{
                        "blue_agents":{
                            "num_agents":2,
                            "base_behavior": "external",       
                            "init_hdg":0.0
                        },
                        "red_agents":{
                            "num_agents":2,
                            "base_behavior": "baseline1",
                            "init_hdg":180.0
                        }
                    }
}

#Update de default configuration with the desired changes
B_ACE_config.update(env_config)
B_ACE_config.update(agents_config)

# Create an instance of the GodotRLPettingZooWrapper
print("Initializing Godot environment...")
# 直接使用原始 Wrapper 测试连接
env = B_ACE_GodotPettingZooWrapper(**B_ACE_config)

# Reset the environment
observations, info = env.reset()

num_steps = 3000
print(f"\nRunning 0 / {num_steps} steps", end = "")

for step in range(num_steps):
    actions = {}
    turn_side = 1
    for agent in env.possible_agents:
        # Action construction
        actions[agent] = np.array([0.1 * turn_side, 0.5, 2.0, 0.0], dtype=np.float32)
        turn_side *= -1
    
    # Take a step
    observations, rewards, termination, truncation, info = env.step(actions)

    # === 修复点 START ===
    # 兼容处理：termination/truncation 可能是 bool (全局) 也可能是 dict (多智能体)
    is_terminated = termination if isinstance(termination, bool) else any(termination.values())
    is_truncated = truncation if isinstance(truncation, bool) else any(truncation.values())

    if is_terminated or is_truncated:
        print(f"\nEpisode finished at step {step}.")
        # 如果需要循环测试，可以在这里 env.reset()
        break 
    # === 修复点 END ===

    if step % 100 == 0:
        print(f'\rRunning {step} / {num_steps} steps', end = "", flush=True)

env.close()
print("\nSimple example script finished.")