from collections import defaultdict
import gymnasium as gym
import numpy as np


# ==== 1. Discretization settings for CartPole ====

# How many bins for each of the 4 observation values
N_BINS = (6, 6, 12, 12)

# Reasonable min/max values for each component
# (we clip to these ranges)
OBS_LOW = np.array([-4.8, -5.0, -0.418, -5.0])
OBS_HIGH = np.array([4.8, 5.0, 0.418, 5.0])


def discretize(obs: np.ndarray) -> tuple[int, int, int, int]:
    """
    Convert continuous observation into a discrete state (tuple of ints).

    obs: [cart_pos, cart_vel, pole_angle, pole_vel]
    """
    # Clip the observation to the allowed range
    clipped = np.clip(obs, OBS_LOW, OBS_HIGH)

    # Compute ratios in [0, 1]
    ratios = (clipped - OBS_LOW) / (OBS_HIGH - OBS_LOW)

    # Scale to [0, N_BINS[i]) and take integer part
    new_obs = (ratios * np.array(N_BINS)).astype(int)

    # Make sure we are inside [0, N_BINS[i] - 1]
    new_obs = np.clip(new_obs, 0, np.array(N_BINS) - 1)

    return tuple(new_obs)
    # Example output: (2, 3, 7, 5)


# ==== 2. Q-learning agent (almost same as your BlackjackAgent) ====

class CartPoleAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.99,
    ):
        self.env = env

        # Q-table: maps (discrete_state, action) -> value
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        # Exploration parameters
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []

    def get_action(self, state) -> int:
        """Epsilon-greedy action from a DISCRETE state."""
        if np.random.random() < self.epsilon:
            # Explore
            return self.env.action_space.sample()
        else:
            # Exploit
            return int(np.argmax(self.q_values[state]))

    def update(self, state, action, reward, terminated, next_state):
        """Standard Q-learning update."""
        future_q_value = (not terminated) * np.max(self.q_values[next_state])
        target = reward + self.discount_factor * future_q_value
        td_error = target - self.q_values[state][action]

        self.q_values[state][action] += self.lr * td_error
        self.training_error.append(td_error)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


# ==== 3. Training loop ====

def train_cartpole():
    env = gym.make("CartPole-v1")  # no render during training (faster)

    n_episodes = 10_000
    learning_rate = 0.1
    start_epsilon = 1.0
    final_epsilon = 0.01
    epsilon_decay = (start_epsilon - final_epsilon) / (n_episodes * 0.8)

    agent = CartPoleAgent(
        env=env,
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
        discount_factor=0.99,
    )

    rewards_per_episode = []

    for episode in range(n_episodes):
        obs, info = env.reset()
        state = discretize(obs)

        terminated = False
        truncated = False
        episode_reward = 0.0

        while not (terminated or truncated):
            action = agent.get_action(state)
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_state = discretize(next_obs)

            agent.update(state, action, reward, terminated, next_state)

            state = next_state
            episode_reward += reward

        agent.decay_epsilon()
        rewards_per_episode.append(episode_reward)

        if (episode + 1) % 1000 == 0:
            last_100_avg = np.mean(rewards_per_episode[-100:])
            print(
                f"Episode {episode + 1}/{n_episodes} | "
                f"epsilon={agent.epsilon:.3f} | "
                f"avg reward (last 100): {last_100_avg:.1f}"
            )

    env.close()
    return agent, rewards_per_episode


# ==== 4. Watch the trained agent ====

def watch_agent(agent: CartPoleAgent, n_episodes: int = 5):
    # New env with render_mode="human" so we can see it
    env = gym.make("CartPole-v1", render_mode="human")

    # Turn off exploration
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0

    for i in range(n_episodes):
        obs, info = env.reset()
        state = discretize(obs)
        terminated = False
        truncated = False
        total_reward = 0

        while not (terminated or truncated):
            action = agent.get_action(state)
            obs, reward, terminated, truncated, info = env.step(action)
            state = discretize(obs)
            total_reward += reward

        print(f"Test episode {i + 1}: total reward = {total_reward}")

    agent.epsilon = old_epsilon
    env.close()


if __name__ == "__main__":
    agent, rewards = train_cartpole()
    watch_agent(agent)
