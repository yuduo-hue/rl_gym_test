import time
import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper

gym.register_envs(ale_py)

env = gym.make("ALE/Pong-v5")
env = AtariWrapper(env)

model = DQN("CnnPolicy", env, verbose=0)  # keep hyperparams same as your real run

N = 50_000  # small test
start = time.time()
model.learn(total_timesteps=N)
elapsed = time.time() - start

steps_per_second = N / elapsed
print(f"Steps per second: {steps_per_second:.1f}")
print(f"Steps per minute: {steps_per_second*60:.0f}")
