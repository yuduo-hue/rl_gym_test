import numpy as np
import gymnasium as gym
from scipy.linalg import solve_discrete_are
from scipy.signal import cont2discrete

def dlqr(Ad, Bd, Q, R):
    P = solve_discrete_are(Ad, Bd, Q, R)
    K = np.linalg.inv(Bd.T @ P @ Bd + R) @ (Bd.T @ P @ Ad)
    return K

# --- Gym default parameters ---
g = 9.8
m_c = 1.0
m_p = 0.1
total_mass = m_c + m_p
l = 0.5                 # half pole length
p = m_p * l             # polemass_length in Gym
tau = 0.02
force_mag = 10.0

# Gym denominator term: l * (4/3 - m_p * cos^2(theta) / total_mass)
# Linearizing at theta=0 => cos=1
den = l * (4.0/3.0 - m_p / total_mass)

# Linearized coefficients (from Gym’s equations around theta=0)
c_theta = g / den
d_force = -1.0 / (total_mass * den)

a_theta = -(p * g) / (total_mass * den)
b_force = (1.0 / total_mass) + (p / (total_mass**2 * den))

# Continuous-time state: [x, x_dot, theta, theta_dot]
A_c = np.array([
    [0, 1,      0, 0],
    [0, 0, a_theta, 0],
    [0, 0,      0, 1],
    [0, 0, c_theta, 0],
], dtype=float)

B_c = np.array([
    [0],
    [b_force],
    [0],
    [d_force],
], dtype=float)

# Discretize with ZOH (better than Euler)
Ad, Bd, _, _, _ = cont2discrete((A_c, B_c, np.eye(4), np.zeros((4,1))), tau)

# Start weights (tune later)
Q = np.diag([10.0, 0.1, 10.0, 0.1])
R = np.array([[0.1]])

K = dlqr(Ad, Bd, Q, R)
print("K =", K)

env = gym.make("CartPole-v1", render_mode="human")
obs, info = env.reset(seed=0)

def policy(obs):
    x = obs.reshape(4, 1)
    u = (-(K @ x)).item()  # continuous "force"
    # map to Gym's discrete action: 0=left, 1=right
    return 1 if u > 0 else 0

total_reward = 0
for t in range(2000):
    action = policy(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    if terminated or truncated:
        break

env.close()
print("Episode length:", t + 1, "Total reward:", total_reward)
