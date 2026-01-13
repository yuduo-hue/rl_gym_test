import gymnasium as gym
from stable_baselines3 import DQN

def main():
    # 1. 创建环境
    env = gym.make("CartPole-v1")

    # 2. 创建 DQN 模型
    model = DQN(
        "MlpPolicy",   # 使用全连接网络（状态是低维向量，不是图像）
        env,
        learning_rate=1e-3,
        buffer_size=50_000,
        learning_starts=1_000,  # 先随机收集一些经验再开始学习
        batch_size=64,
        tau=1.0,                 # 软更新系数
        gamma=0.99,              # 折扣因子
        train_freq=4,            # 每 4 步更新一次
        target_update_interval=1_000,  # 多久更新一次目标网络
        verbose=1,               # 输出训练日志
    )

    # 3. 训练
    model.learn(total_timesteps=1_400_000)

    # 4. 保存模型（可选）
    model.save("dqn_cartpole_sb3")

    # 5. 测试训练好的模型
    test_trained_model(model)


def test_trained_model(model, n_episodes: int = 5):
    env = gym.make("CartPole-v1", render_mode="human")

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            # 使用确定性策略（不再探索）
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        print(f"Episode {ep + 1}: total reward = {total_reward}")

    env.close()


if __name__ == "__main__":
    main()
