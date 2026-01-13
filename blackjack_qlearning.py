from collections import defaultdict
import gymnasium as gym
import numpy as np


class BlackjackAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        """Initialize a Q-Learning agent.

        Args:
            env: The training environment
            learning_rate: How quickly to update Q-values (0-1)
            initial_epsilon: Starting exploration rate (usually 1.0)
            epsilon_decay: How much to reduce epsilon each episode
            final_epsilon: Minimum exploration rate (usually 0.1)
            discount_factor: How much to value future rewards (0-1)
        """
        self.env = env

        # Q-table: maps (state, action) to expected reward
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor  # How much we care about future rewards

        # Exploration parameters
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        # Track learning progress
        self.training_error = []

    def get_action(self, obs: tuple[int, int, bool]) -> int:
        """Choose an action using epsilon-greedy strategy.

        Returns:
            action: 0 (stand) or 1 (hit)
        """
        # With probability epsilon: explore (random action)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()

        # With probability (1-epsilon): exploit (best known action)
        else:
            return int(np.argmax(self.q_values[obs]))

    def update(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
    ):
        """Update Q-value based on experience."""
        # Best we can do from next state (0 if episode ended)
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])

        # Target = immediate reward + discounted future best
        target = reward + self.discount_factor * future_q_value

        # Error between target and our current estimate
        temporal_difference = target - self.q_values[obs][action]

        # Move Q-value a bit toward the target
        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )

        # Save error for analysis later
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        """Reduce exploration rate after each episode."""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

def train_blackjack():
    # Training hyperparameters (from tutorial)
    learning_rate = 0.01        # How fast to learn
    n_episodes = 100_000        # How many hands to play
    start_epsilon = 1.0         # Start fully random
    epsilon_decay = start_epsilon / (n_episodes / 2)  # Slowly reduce randomness
    final_epsilon = 0.1         # Keep some exploration forever

    # Create Blackjack environment
    env = gym.make("Blackjack-v1", sab=False)

    # Wrap env to record returns/lengths if you want later analysis (optional)
    # from gymnasium.wrappers import RecordEpisodeStatistics
    # env = RecordEpisodeStatistics(env, buffer_length=n_episodes)

    # Create agent
    agent = BlackjackAgent(
        env=env,
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
    )

    # Training loop
    for episode in range(n_episodes):
        obs, info = env.reset()
        terminated = False
        truncated = False

        while not (terminated or truncated):
            # 1. Agent chooses action
            action = agent.get_action(obs)

            # 2. Environment responds
            next_obs, reward, terminated, truncated, info = env.step(action)

            # 3. Agent learns from this experience
            agent.update(obs, action, reward, terminated, next_obs)

            # 4. Move to next state
            obs = next_obs

        # After each episode, reduce exploration a bit
        agent.decay_epsilon()

        # (Optional) print progress every 10,000 episodes
        if (episode + 1) % 10_000 == 0:
            print(f"Finished episode {episode + 1}/{n_episodes}")

    return agent, env

def test_agent(agent: BlackjackAgent, env: gym.Env, num_episodes: int = 1000):
    """Test the trained agent without exploration."""
    total_rewards = []

    # Save and disable exploration
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0  # always choose best action

    for _ in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = agent.get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        total_rewards.append(episode_reward)

    # Restore old epsilon
    agent.epsilon = old_epsilon

    avg_reward = np.mean(total_rewards)
    win_rate = np.mean(np.array(total_rewards) > 0)

    print(f"Test over {num_episodes} episodes:")
    print(f"Average reward: {avg_reward:.3f}")
    print(f"Win rate: {win_rate:.1%}")

if __name__ == "__main__":
    agent, env = train_blackjack()
    test_agent(agent, env)
