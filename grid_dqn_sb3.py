from stable_baselines3 import DQN
from simple_grid_env import SimpleGridEnv

env = SimpleGridEnv()
model = DQN("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50_000)