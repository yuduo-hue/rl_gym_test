import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame


class SimplePlatformerEnv(gym.Env):
    """
    A small 2D platformer-like environment:

    - Player is a square that can move left/right and jump.
    - Goal is a square on the right side.
    - Observation: 84x84 grayscale image (H, W, 1).
    - Actions:
        0: idle
        1: move left
        2: move right
        3: jump
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode: str | None = None):
        super().__init__()

        # --- Game world parameters ---
        self.window_width = 160
        self.window_height = 120
        self.player_size = 10
        self.goal_size = 10

        # Physics
        self.gravity = 0.5
        self.move_speed = 2.0
        self.jump_speed = -7.0

        # Player state
        self.player_x = None
        self.player_y = None
        self.player_vx = 0.0
        self.player_vy = 0.0
        self.on_ground = False

        # Goal position (fixed near the right side)
        self.goal_x = self.window_width - 20
        self.goal_y = self.window_height - self.goal_size - 5

        # Gym spaces
        # Observation: 84x84 grayscale image
        self.observation_shape = (84, 84, 1)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=self.observation_shape,
            dtype=np.uint8,
        )

        # Actions: 0 idle, 1 left, 2 right, 3 jump
        self.action_space = spaces.Discrete(4)

        # Render stuff
        self.render_mode = render_mode
        pygame.init()
        self.canvas = pygame.Surface((self.window_width, self.window_height))
        self.display = None
        if self.render_mode == "human":
            self.display = pygame.display.set_mode(
                (self.window_width, self.window_height)
            )
            pygame.display.set_caption("Simple Platformer Env")

        self.clock = pygame.time.Clock()
        self.step_count = 0
        self.max_steps = 500  # episode time limit

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initial player position: near the left, on the floor
        self.player_x = 20.0
        self.player_y = self.window_height - self.player_size - 5
        self.player_vx = 0.0
        self.player_vy = 0.0
        self.on_ground = True

        self.step_count = 0

        # Clear events to avoid window "not responding"
        pygame.event.pump()

        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action: int):
        # Handle window close events (important!)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

        self.step_count += 1

        # --- Apply action to player velocity ---
        if action == 1:  # left
            self.player_vx = -self.move_speed
        elif action == 2:  # right
            self.player_vx = self.move_speed
        else:
            # small friction if idle or jump
            self.player_vx = 0.0

        if action == 3 and self.on_ground:
            self.player_vy = self.jump_speed
            self.on_ground = False

        # --- Update physics ---
        # Gravity
        self.player_vy += self.gravity
        
        old_dist = abs(self.goal_x - self.player_x)
        
        # Update position
        self.player_x += self.player_vx
        self.player_y += self.player_vy
        
        new_dist = abs(self.goal_x - self.player_x)

        # Floor collision
        floor_y = self.window_height - self.player_size - 5
        if self.player_y >= floor_y:
            self.player_y = floor_y
            self.player_vy = 0.0
            self.on_ground = True

        # Walls: keep inside [0, width - player_size]
        self.player_x = np.clip(self.player_x, 0, self.window_width - self.player_size)

        # --- Check goal collision ---
        terminated = False
        truncated = False

        if self._collides_with_goal():
            terminated = True

        if self.step_count >= self.max_steps:
            truncated = True

        # --- Reward function (shaped) ---
        # Dense part: change in distance (moving closer is good)
        dist_reward = (old_dist - new_dist) * 1  # scale factor
        
        # Terminal bonus
        if terminated:
            reward = 100.0 + dist_reward
        else:
            reward = dist_reward  # small step cost

        obs = self._get_obs()

        if self.render_mode == "human":
            self._render_frame()

        info = {}
        return obs, reward, terminated, truncated, info

    def _collides_with_goal(self) -> bool:
        # Axis-aligned bounding box check
        px1 = self.player_x
        py1 = self.player_y
        px2 = px1 + self.player_size
        py2 = py1 + self.player_size

        gx1 = self.goal_x
        gy1 = self.goal_y
        gx2 = gx1 + self.goal_size
        gy2 = gy1 + self.goal_size

        overlap_x = (px1 < gx2) and (px2 > gx1)
        overlap_y = (py1 < gy2) and (py2 > gy1)
        return overlap_x and overlap_y

    def _draw_world(self):
        # Fill background
        self.canvas.fill((30, 30, 30))

        # Draw floor
        floor_rect = pygame.Rect(0, self.window_height - 5, self.window_width, 5)
        pygame.draw.rect(self.canvas, (50, 50, 50), floor_rect)

        # Draw goal
        goal_rect = pygame.Rect(
            int(self.goal_x),
            int(self.goal_y),
            self.goal_size,
            self.goal_size,
        )
        pygame.draw.rect(self.canvas, (0, 200, 0), goal_rect)

        # Draw player
        player_rect = pygame.Rect(
            int(self.player_x),
            int(self.player_y),
            self.player_size,
            self.player_size,
        )
        pygame.draw.rect(self.canvas, (200, 200, 0), player_rect)

    def _get_obs(self):
        # Draw the scene to the canvas
        self._draw_world()

        # Resize to 84x84 and convert to grayscale
        small_surf = pygame.transform.smoothscale(self.canvas, (84, 84))
        # Pygame surfarray returns (width, height, 3); we transpose later
        arr = pygame.surfarray.array3d(small_surf).astype(np.float32)
        arr = np.transpose(arr, (1, 0, 2))  # (H, W, C)

        # Convert to grayscale: simple mean over channels
        gray = arr.mean(axis=2)  # (H, W)
        gray = gray.astype(np.uint8)

        # Add channel dimension: (H, W, 1)
        obs = gray[..., None]
        return obs

    def _render_frame(self):
        if self.display is None:
            self.display = pygame.display.set_mode(
                (self.window_width, self.window_height)
            )
        self.display.blit(self.canvas, (0, 0))
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def render(self):
        # For Gymnasium's render() API
        if self.render_mode == "human":
            self._render_frame()
        elif self.render_mode == "rgb_array":
            self._draw_world()
            small_surf = pygame.transform.smoothscale(self.canvas, (84, 84))
            arr = pygame.surfarray.array3d(small_surf).astype(np.uint8)
            arr = np.transpose(arr, (1, 0, 2))
            return arr

    def close(self):
        pygame.quit()
