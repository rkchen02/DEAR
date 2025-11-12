import numpy as np
import gymnasium as gym
from gymnasium import spaces


class BellmanFordEnv(gym.Env):
    """
    Bellman-Ford environment as a MDP with CLRS-like evaluation helpers.

    Termination (classical BF style):
      - The environment tracks which edges have been visited in the current
        'pass' (a pass is complete once ALL edges have been selected at least
        once). After a full pass, if no relaxation happened during that pass,
        the episode terminates (converged). Otherwise we start a new pass.
      - Also terminate if the number of completed passes >= (n_nodes - 1).

    Action space:
      - Discrete(n_edges) where edges are enumerated as (u, v) for u != v.

    Observation:
      Dict with:
        "d": distances (float32 vector length n_nodes)
        "pi": predecessors (int32 vector length n_nodes)
        "w": adjacency weight matrix (float32 n_nodes x n_nodes)
    """

    metadata = {"render.modes": []}

    def __init__(self, n_nodes=5, max_steps=500, reward_mode="dense", seed=None):
        super().__init__()
        self.n_nodes = int(n_nodes)
        self.max_nodes = 12
        self.max_steps = int(max_steps)
        self.reward_mode = reward_mode

        # Build canonical edge list (directed) and map index -> (u,v)
        self.edges = [(u, v) for u in range(self.n_nodes) for v in range(self.n_nodes) if u != v]
        self.n_edges = len(self.edges)

        # Action space: pick one directed edge index
        self.action_space = spaces.Discrete(self.n_edges)

        # Observation space (dict)
        self.observation_space = spaces.Dict({
            "d": spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_nodes,), dtype=np.float32),
            "pi": spaces.Box(low=-1, high=self.max_nodes - 1, shape=(self.max_nodes,), dtype=np.int32),
            "w": spaces.Box(low=-1.0, high=1.0, shape=(self.max_nodes, self.max_nodes), dtype=np.float32),
        })


        # RNG
        self.np_random, _ = gym.utils.seeding.np_random(seed)

        # initialise graph + ground-truth (set in reset)
        self._init_graph()
        self.reset()

    def _init_graph(self):
        """Create random directed weights in [-1, 1], zero on diagonal."""
        w = self.np_random.uniform(-1.0, 1.0, (self.n_nodes, self.n_nodes)).astype(np.float32)
        np.fill_diagonal(w, 0.0)

        # Pad to max_nodes x max_nodes
        if self.n_nodes < self.max_nodes:
            pad = self.max_nodes - self.n_nodes
            w = np.pad(w, ((0, pad), (0, pad)), mode="constant", constant_values=0.0)

        self.w = w

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
        """
        Reset environment:
         - generate fresh random graph weights
         - compute ground-truth by classical Bellman-Ford (for evaluation)
         - initialize distance vector with +inf except source 0
         - reset pass bookkeeping
        """
        super().reset(seed=seed)
        self._init_graph()
        self.src = 0

        # agent's current estimates
        self.d = np.full(self.n_nodes, np.inf, dtype=np.float32)
        self.d[self.src] = 0.0
        self.pi = np.full(self.n_nodes, -1, dtype=np.int32)

        # bookkeeping for passes (classical BF semantics)
        self.edge_visited = np.zeros(self.n_edges, dtype=bool)  # visited mask for current pass
        self.pass_updated = False  # whether any relaxation occurred in current pass
        self.completed_passes = 0   # completed passes so far
        self.steps = 0
        self.done_flag = False

        # Precompute ground-truth distances and predecessors (classical BF)
        self.gt_d, self.gt_pi = self._compute_ground_truth()

        return self._get_obs(), {}

    def step(self, action):
        """
        action: integer index into self.edges
        returns: obs, reward, done, truncated, info
        """
        if self.done_flag:
            raise RuntimeError("Episode is done. Call reset() before step().")

        if isinstance(action, (np.ndarray, list, tuple)):
            action = int(np.asarray(action).item())
        action = int(action)
        if action < 0 or action >= self.n_edges:
            raise ValueError("Invalid action index")

        u, v = self.edges[action]
        self.steps += 1

        # Perform relaxation for chosen edge
        new_dv = self.d[u] + self.w[u, v]
        improved = False
        if new_dv < self.d[v]:
            self.d[v] = new_dv
            self.pi[v] = u
            improved = True
            self.pass_updated = True

        # Mark edge visited in this pass
        self.edge_visited[action] = True

        # Reward
        if self.reward_mode == "dense":
            reward = 1.0 if improved else 0.0
        elif self.reward_mode == "sparse":
            reward = 0.0
        else:
            raise ValueError(f"Invalid reward_mode: {self.reward_mode}")

        # If all edges have been visited => end of a full pass
        if self.edge_visited.all():
            # If no improvement in the entire pass => converged (classical BF early stop)
            if not self.pass_updated:
                done = True
                reason = "converged_no_updates"
            else:
                # completed a pass with at least one update
                self.completed_passes += 1
                # classical BF requires at most (n_nodes - 1) passes
                if self.completed_passes >= (self.n_nodes - 1):
                    done = True
                    reason = "max_passes_reached"
                else:
                    # start a new pass
                    done = False
                    reason = "start_new_pass"
                # reset pass bookkeeping
                self.edge_visited[:] = False
                self.pass_updated = False
        else:
            done = False
            reason = "pass_incomplete"

        # Also terminate if step budget exceeded
        if self.steps >= self.max_steps:
            done = True
            reason = "max_steps"

        self.done_flag = bool(done)
        self.prev_d = self.d.copy()  # keep for potential older logic

        obs = self._get_obs()
        info = {"reason": reason}
        return obs, float(reward), bool(done), False, info

    def _compute_ground_truth(self):
        """Classical Bellman-Ford to get true distances and predecessors."""
        n = self.n_nodes
        inf = float("inf")
        d = [inf] * n
        pi = [-1] * n
        d[self.src] = 0.0

        # relax all edges up to n-1 times (deterministic full passes)
        for _ in range(n - 1):
            updated = False
            for (u, v) in self.edges:
                if d[u] == inf:
                    continue
                nd = d[u] + float(self.w[u, v])
                if nd < d[v]:
                    d[v] = nd
                    pi[v] = u
                    updated = True
            if not updated:
                break

        # convert to numpy arrays (float32 / int32)
        d_arr = np.array(d, dtype=np.float32)
        pi_arr = np.array(pi, dtype=np.int32)
        return d_arr, pi_arr

    def compute_metrics(self):
        """
        Return CLRS-style metrics comparing current agent estimate to Bellman-Ford ground truth.

        Returns:
            dict with:
                - distance_accuracy: fraction of nodes where d ≈ gt_d
                - pointer_accuracy: fraction of nodes where pi == gt_pi
                - distance_match_vector: boolean mask over nodes
                - pointer_match_vector: boolean mask over nodes
        """
        # Ensure ground truth exists
        if not hasattr(self, "gt_d") or not hasattr(self, "gt_pi"):
            raise AttributeError("Ground-truth values (gt_d, gt_pi) not set in environment.")

        gt_d = np.asarray(self.gt_d, dtype=np.float32)
        gt_pi = np.asarray(self.gt_pi, dtype=np.int32)
        d = np.asarray(self.d, dtype=np.float32)
        pi = np.asarray(self.pi, dtype=np.int32)

        # Ensure matching length (important for padded graphs)
        n = min(len(gt_d), len(d))

        dist_match = np.zeros(n, dtype=bool)
        tol = 1e-4
        for i in range(n):
            if np.isinf(gt_d[i]) and np.isinf(d[i]):
                dist_match[i] = True
            elif np.isfinite(gt_d[i]) and np.isfinite(d[i]):
                dist_match[i] = np.isclose(gt_d[i], d[i], atol=tol, rtol=0.0)
            else:
                dist_match[i] = False

        pointer_match = (pi[:n] == gt_pi[:n])

        distance_accuracy = dist_match.mean() if n > 0 else 0.0
        pointer_accuracy = pointer_match.mean() if n > 0 else 0.0

        return {
            "distance_accuracy": float(distance_accuracy),
            "pointer_accuracy": float(pointer_accuracy),
            "distance_match_vector": dist_match,
            "pointer_match_vector": pointer_match,
        }


    def render(self):
        print("Current distances:", np.round(self.d, 6))
        print("Predecessors:", self.pi)
        print("Edges shape:", (self.n_edges, 2))
        print("Node features shape:", (self.n_nodes, 1))
        print("Edge features shape:", (self.n_edges, 1))
        print("Adjacency matrix:\n", np.round(self.w, 6))
