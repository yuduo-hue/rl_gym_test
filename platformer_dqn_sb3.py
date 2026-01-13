import gymnasium as gym
from stable_baselines3 import DQN
from simple_platformer_env import SimplePlatformerEnv


def main():
    # Training env (no render for speed)
    env = SimplePlatformerEnv(render_mode=None)

    model = DQN(
    "CnnPolicy",
    env,
    learning_rate=1e-4,
    buffer_size=50_000,
    learning_starts=1_000,
    batch_size=32,
    gamma=0.99,
    train_freq=4,
    target_update_interval=1_000,

    exploration_initial_eps=1.0,   # start very exploratory
    exploration_final_eps=0.05,    # keep a bit of randomness at the end
    exploration_fraction=0.3,      # decay epsilon over first 30% of training

    verbose=1,
)


    model.learn(total_timesteps=200_000)
    model.save("dqn_simple_platformer")

    # Watch the trained agent
    watch_trained_agent(model)


def watch_trained_agent(model, n_episodes: int = 5):
    env = SimplePlatformerEnv(render_mode="human")

    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        print(f"Episode {ep + 1}: total reward = {total_reward}")

    env.close()


if __name__ == "__main__":
    main()
