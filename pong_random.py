import gymnasium as gym
import ale_py

# Register ALE environments with Gymnasium
gym.register_envs(ale_py)

env = gym.make("ALE/Pong-v5")  # no need for render yet
obs, info = env.reset()

print("Observation shape:", obs.shape)
print("Action space:", env.action_space)

for _ in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print("Reward:", reward, "Terminated:", terminated, "Truncated:", truncated)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
