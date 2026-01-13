import gymnasium as gym
import ale_py                               # NEW

from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper

# Register ALE environments with Gymnasium
gym.register_envs(ale_py)                   # NEW


def make_env():
    # raw env
    env = gym.make("ALE/Pong-v5")
    # wrap it with standard Atari preprocessing:
    # grayscale, resize, frame stacking, etc.
    env = AtariWrapper(env)
    return env


def main():
    env = make_env()

    model = DQN(
        "CnnPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=100_000,
        learning_starts=10_000,
        batch_size=32,
        gamma=0.99,
        train_freq=4,
        target_update_interval=10_000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        verbose=1,
    )

    # Training Pong to a strong level usually needs millions of steps.
    # Start with a small number just to check everything runs.
    model.learn(total_timesteps=1_400_000)

    model.save("dqn_pong_sb3")

    # Watch the trained agent
    watch_trained_agent(model)


def watch_trained_agent(model, n_episodes: int = 5):
    # For rendering we make a fresh env with human render
    env = gym.make("ALE/Pong-v5", render_mode="human")
    env = AtariWrapper(env)

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        print(f"Episode {ep + 1}: total reward = {episode_reward}")

    env.close()


if __name__ == "__main__":
    main()
