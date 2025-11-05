import numpy as np
import gymnasium as gym
from gymnasium import spaces


class BellmanFordEnv(gym.Env):
    """
    Bellman-Ford environment as a MDP.

    State (s_t):
        - Node distance estimates (d)
        - Predecessor pointers (pi)
        - Edge weights (w)
    Action (a_t):
        - (u, v) edge to relax
    Reward (r_t):
        - +1 if d[v] was updated, 0 otherwise
    Episode ends when no update is possible or after max_steps.

    Observation:
        Dict containing:
            "d": np.ndarray of shape [n_nodes]
            "pi": np.ndarray of shape [n_nodes]
            "w": np.ndarray of shape [n_nodes, n_nodes]
    """

    metadata = {"render.modes": []}

    def __init__(self, n_nodes=5, max_steps=50, reward_mode="dense", seed=None):
        super().__init__()
        self.n_nodes = n_nodes
        self.max_steps = max_steps
        self.reward_mode = reward_mode

        # Action space: select an edge (u, v)
        self.action_space = spaces.MultiDiscrete([n_nodes, n_nodes])

        # Observation space: node distances, predecessor, weights
        self.observation_space = spaces.Dict({
            "d": spaces.Box(low=-np.inf, high=np.inf, shape=(n_nodes,), dtype=np.float32),
            "pi": spaces.Box(low=-1, high=n_nodes-1, shape=(n_nodes,), dtype=np.int32),
            "w": spaces.Box(low=-1.0, high=1.0, shape=(n_nodes, n_nodes), dtype=np.float32),
        })

        self.np_random, _ = gym.utils.seeding.np_random(seed)
        self._init_graph()
        self.reset()

    def _init_graph(self):
        """Initialize a random weighted directed graph."""
        w = self.np_random.uniform(-1.0, 1.0, (self.n_nodes, self.n_nodes)).astype(np.float32)
        np.fill_diagonal(w, 0.0)
        self.w = w

    def _get_obs(self):
        return {
            "d": np.nan_to_num(self.d.copy(), nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32),
            "pi": self.pi.copy().astype(np.int32),
            "w": np.nan_to_num(self.w.copy(), nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32),
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._init_graph()
        self.src = 0
        self.d = np.full(self.n_nodes, np.inf, dtype=np.float32)
        self.d[self.src] = 0.0
        self.pi = np.full(self.n_nodes, -1, dtype=np.int32)
        self.prev_d = self.d.copy()
        self.steps = 0
        self.done_flag = False
        return self._get_obs(), {}

    def step(self, action):
        if self.done_flag:
            raise RuntimeError("Episode is done. Call reset() before step().")

        u, v = int(action[0]), int(action[1])
        self.steps += 1

        # Apply Bellman-Ford relaxation
        new_dv = self.d[u] + self.w[u, v]
        improved = False

        if new_dv < self.d[v]:
            self.d[v] = new_dv
            self.pi[v] = u
            improved = True

        # Reward function
        if self.reward_mode == "dense":
            reward = 1.0 if improved else 0.0
        elif self.reward_mode == "sparse":
            reward = 0.0
        else:
            raise ValueError(f"Invalid reward_mode: {self.reward_mode}")

        # Check convergence
        converged = np.allclose(self.d, self.prev_d)
        done = converged or self.steps >= self.max_steps
        self.done_flag = done

        if self.reward_mode == "sparse" and done and not np.isinf(self.d).any():
            reward = 1.0

        self.prev_d = self.d.copy()
        obs = self._get_obs()

        return obs, reward, done, False, {}

    def render(self):
        print("Current distances:", self.d)
        print("Predecessors:", self.pi)
        print("Edge weights:\n", self.w)
