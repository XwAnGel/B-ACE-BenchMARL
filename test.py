import torch
import os

# --- 请确认路径是否正确 ---
# 注意：Windows路径中的反斜杠 \ 需要转义，或者在字符串前加 r
model_path = r"E:\angel\B-ACE-main\Results\mappo_b_ace_mlp__193b6b1a_25_12_16-18_09_47\checkpoints\checkpoint_600000.pt"

def analyze_mappo_checkpoint(path):
    if not os.path.exists(path):
        print(f"❌ 错误：找不到文件 {path}")
        return

    print(f"--- 正在分析 MAPPO 模型: {os.path.basename(path)} ---")
    
    try:
        # 加载 checkpoint (映射到 CPU 以免报错)
        checkpoint = torch.load(path, map_location="cpu")
        
        # BenchMARL 的 MAPPO 参数通常存储在类似 'module' 或 'agent' 的结构中
        # 我们需要递归查找特定的权重矩阵
        
        actor_input_dim = None
        critic_input_dim = None
        action_dim = None
        
        print("\n--- 🔍 神经网络层级结构探测 ---")

        def search_weights(d, prefix=""):
            nonlocal actor_input_dim, critic_input_dim, action_dim
            
            if isinstance(d, dict):
                for k, v in d.items():
                    search_weights(v, prefix + k + ".")
            elif isinstance(d, torch.Tensor):
                # 我们假设隐藏层大小是 256 (基于之前的配置)
                # 权重形状通常是 [Output_Features, Input_Features]
                shape = d.shape
                
                if len(shape) == 2:
                    # 1. 寻找 Actor (策略网络) 的输入层
                    # 特征：输出是256，名字里通常带 'logits' 或 'actor' 或位于结构前部
                    # 在 BenchMARL 中，Agent 的网络通常在最外层或 'agent' 下
                    if shape[0] == 256 and "critic" not in prefix and "value" not in prefix:
                        # 这是一个简化的启发式判断：如果没找到过 Actor 输入，且不是 Critic，且输出是 256
                        if actor_input_dim is None: 
                            print(f"👉 发现疑似 [Actor/策略] 输入层: '{prefix}weight' | 形状 {shape}")
                            actor_input_dim = shape[1]

                    # 2. 寻找 Critic (价值网络) 的输入层
                    # 特征：输出是 256 (因为第一层隐藏层通常也是256或512)，名字里带 'critic' 或 'value'
                    # 注意：您的配置里 critic 是 [512, 256, 256]，所以第一层输出可能是 512
                    elif (shape[0] == 256 or shape[0] == 512) and ("critic" in prefix or "value" in prefix):
                        if critic_input_dim is None:
                            print(f"👉 发现疑似 [Critic/价值] 输入层: '{prefix}weight' | 形状 {shape}")
                            critic_input_dim = shape[1]
                            
                    # 3. 寻找输出层 (动作)
                    # 输入是 256，输出很小 (比如 4, 8, 10)
                    elif shape[1] == 256 and shape[0] < 50:
                        print(f"👉 发现疑似 [输出层] 权重: '{prefix}weight' | 形状 {shape}")
                        if "critic" not in prefix and "value" not in prefix:
                            action_dim = shape[0]

        # 开始递归搜索
        search_weights(checkpoint)
        
        print("\n--- 📊 分析结论 ---")
        if actor_input_dim:
            print(f"1. 观测维度 (Observation Dim): {actor_input_dim}")
            print("   (这是每架飞机自己能看到的数据量)")
        else:
            print("1. 未能自动识别 Actor 输入维度")

        if critic_input_dim:
            print(f"2. 全局状态维度 (Global State Dim): {critic_input_dim}")
            print("   (这是 Critic 看到的全局信息量)")
        else:
            print("2. 未能自动识别 Critic 输入维度")

        if actor_input_dim and critic_input_dim:
            if critic_input_dim > actor_input_dim:
                print(f"\n✅ MAPPO 特征确认：Critic 输入 ({critic_input_dim}) > Actor 输入 ({actor_input_dim})")
                print("   说明 Critic 确实利用了额外的全局信息！")
            elif critic_input_dim == actor_input_dim:
                print(f"\n⚠️ 注意：Critic 输入 等于 Actor 输入。")
                print("   这可能意味着使用的是 IPPO 模式，或者全局信息和局部观测恰好大小一致。")
            
    except Exception as e:
        print(f"读取出错: {e}")

if __name__ == "__main__":
    analyze_mappo_checkpoint(model_path)