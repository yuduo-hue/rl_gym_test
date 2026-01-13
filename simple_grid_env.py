import gymnasium as gym
from gymnasium import spaces
import numpy as np

class SimpleGridEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None):
        super().__init__()
        self.size = 5
        self.observation_space = spaces.Discrete(self.size * self.size)
        self.action_space = spaces.Discrete(4)  # 0: up, 1: right, 2: down, 3: left
        self.state = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # bottom-left index
        self.state = (self.size - 1) * self.size + 0
        return self.state, {}

    def step(self, action):
        # convert index to (row, col)
        row, col = divmod(self.state, self.size)

        if action == 0 and row > 0:
            row -= 1
        elif action == 1 and col < self.size - 1:
            col += 1
        elif action == 2 and row < self.size - 1:
            row += 1
        elif action == 3 and col > 0:
            col -= 1

        self.state = row * self.size + col

        # goal at top-right
        goal_state = 0 * self.size + (self.size - 1)
        terminated = self.state == goal_state
        truncated = False

        reward = 10.0 if terminated else -1.0

        return self.state, reward, terminated, truncated, {}

    def render(self):
        grid = np.full((self.size, self.size), ".")
        r, c = divmod(self.state, self.size)
        grid[r, c] = "A"
        print("\n".join(" ".join(row) for row in grid))
        print()
