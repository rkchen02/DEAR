import numpy as np
import gymnasium as gym
from gymnasium import spaces


class BellmanFordEnv(gym.Env):
    """
    Bellman-Ford shortest-path algorithm framed as an MDP.

    State (s_t):
        - d: current distance estimates (float vector)
        - pi: predecessor pointers (int vector)
        - w: adjacency matrix of edge weights

    Action (a_t):
        - (u, v): choose an edge to relax

    Reward (r_t):
        - +1 if d[v] was updated (successful relaxation)
        - 0 otherwise

    Episode ends:
        - When no updates occur in an entire pass (convergence)
        - Or after n_nodes - 1 relaxation rounds (classical Bellman–Ford termination)
    """

    metadata = {"render.modes": []}

    def __init__(self, n_nodes=5, max_steps=None, reward_mode="dense", seed=None):
        super().__init__()
        self.n_nodes = n_nodes
        self.max_nodes = 12
        self.max_steps = max_steps if max_steps is not None else n_nodes * (n_nodes - 1)
        self.reward_mode = reward_mode

        # Spaces
        self.action_space = spaces.MultiDiscrete([n_nodes, n_nodes])
        self.observation_space = spaces.Dict({
            "d": spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_nodes,), dtype=np.float32),
            "pi": spaces.Box(low=-1, high=self.max_nodes - 1, shape=(self.max_nodes,), dtype=np.int32),
            "w": spaces.Box(low=-1.0, high=1.0, shape=(self.max_nodes, self.max_nodes), dtype=np.float32),
        })

        self.np_random, _ = gym.utils.seeding.np_random(seed)
        self._init_graph()
        self.reset()

    def _init_graph(self):
        """Initialize a random weighted directed graph."""
        w = self.np_random.uniform(-1.0, 1.0, (self.n_nodes, self.n_nodes)).astype(np.float32)
        np.fill_diagonal(w, 0.0)

        # Pad to max_nodes x max_nodes
        if self.n_nodes < self.max_nodes:
            pad = self.max_nodes - self.n_nodes
            w = np.pad(w, ((0, pad), (0, pad)), mode="constant", constant_values=0.0)

        self.w = w

    def _compute_ground_truth(self):
        """Compute ground truth shortest distances and predecessors via classical Bellman-Ford."""
        n = self.n_nodes
        src = 0
        dist = np.full(n, np.inf, dtype=np.float32)
        pred = np.full(n, -1, dtype=np.int32)
        dist[src] = 0.0

        # Relax edges up to n - 1 times
        for _ in range(n - 1):
            updated = False
            for u in range(n):
                for v in range(n):
                    new_dist = dist[u] + self.w[u, v]
                    if new_dist < dist[v]:
                        dist[v] = new_dist
                        pred[v] = u
                        updated = True
            if not updated:
                break

        self.gt_d = dist
        self.gt_pi = pred

    def _get_obs(self):
        d = np.nan_to_num(self.d.copy(), nan=0.0, posinf=1e6, neginf=-1e6)
        pi = self.pi.copy()

        # Pad to fixed length
        if len(d) < self.max_nodes:
            pad = self.max_nodes - len(d)
            d = np.pad(d, (0, pad), constant_values=0.0)
            pi = np.pad(pi, (0, pad), constant_values=-1)

        return {
            "d": d.astype(np.float32),
            "pi": pi.astype(np.int32),
            "w": np.nan_to_num(self.w.copy(), nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32),
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._init_graph()
        self._compute_ground_truth()

        self.src = 0
        self.d = np.full(self.n_nodes, np.inf, dtype=np.float32)
        self.d[self.src] = 0.0
        self.pi = np.full(self.n_nodes, -1, dtype=np.int32)

        self.steps = 0
        self.relax_passes = 0
        self.updated_in_pass = False
        self.done_flag = False

        return self._get_obs(), {}

    def step(self, action):
        if self.done_flag:
            raise RuntimeError("Episode is done. Call reset() before step().")

        u, v = int(action[0]), int(action[1])
        u = np.clip(u, 0, self.n_nodes - 1)
        v = np.clip(v, 0, self.n_nodes - 1)
        self.steps += 1

        reward = 0.0
        improved = False

        if np.isfinite(self.d[u]):
            new_dv = self.d[u] + self.w[u, v]
            if new_dv < self.d[v]:
                self.d[v] = new_dv
                self.pi[v] = u
                improved = True
                reward = 1.0

        # Track updates in this pass
        self.updated_in_pass = self.updated_in_pass or improved

        # One pass = n_nodes edge relaxations
        if self.steps % (self.n_nodes * (self.n_nodes - 1)) == 0:
            self.relax_passes += 1
            if not self.updated_in_pass or self.relax_passes >= (self.n_nodes - 1):
                self.done_flag = True
            self.updated_in_pass = False

        done = self.done_flag or self.steps >= self.max_steps
        obs = self._get_obs()
        info = {"reason": "pass_incomplete" if not done else "terminated"}

        return obs, reward, done, False, info

    def compute_metrics(self):
        """CLRS-style accuracy metrics comparing d, pi to ground truth."""
        gt_d = self.gt_d
        gt_pi = self.gt_pi
        d = self.d
        tol = 1e-4

        dist_match = np.zeros(self.n_nodes, dtype=bool)
        for i in range(self.n_nodes):
            if np.isinf(gt_d[i]) and np.isinf(d[i]):
                dist_match[i] = True
            elif np.isfinite(gt_d[i]) and np.isfinite(d[i]):
                dist_match[i] = np.isclose(gt_d[i], d[i], atol=tol, rtol=0.0)

        pointer_match = (self.pi == gt_pi)

        return {
            "distance_accuracy": float(dist_match.mean()),
            "pointer_accuracy": float(pointer_match.mean()),
            "distance_match_vector": dist_match,
            "pointer_match_vector": pointer_match,
        }

    def render(self):
        print("Current distances:", np.round(self.d, 3))
        print("Predecessors:", self.pi)
        print("Edge weights:\n", np.round(self.w, 3))
        print("Ground truth distances:", np.round(self.gt_d, 3))
        print("Ground truth preds:", self.gt_pi)
