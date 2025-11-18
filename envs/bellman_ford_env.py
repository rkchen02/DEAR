import numpy as np
import gymnasium as gym
from gymnasium import spaces
import numpy as np

def compute_ground_truth(A: np.ndarray, s: int):
    """Compute Bellman–Ford ground truth distances and predecessors (CLRS-style).

    This is adapted from clrs/_src/algorithms/graphs.py: bellman_ford(),
    but stripped of probing, chex, and TensorFlow dependencies.
    """
    n = A.shape[0]
    d = np.full(n, np.inf, dtype=np.float32)
    pi = np.full(n, -1, dtype=np.int32)
    mask = np.zeros(n, dtype=np.bool_)

    d[s] = 0.0
    mask[s] = True

    while True:
        prev_d = d.copy()
        prev_mask = mask.copy()

        for u in range(n):
            if not prev_mask[u]:
                continue
            for v in range(n):
                if A[u, v] != 0:
                    if not mask[v] or prev_d[u] + A[u, v] < d[v]:
                        d[v] = prev_d[u] + A[u, v]
                        pi[v] = u
                    mask[v] = True

        if np.allclose(d, prev_d):
            break

    return d, pi


class BellmanFordEnv(gym.Env):
    """
    Bellman-Ford shortest-path algorithm framed as an MDP.
    Uses CLRS dataset graphs.
    """

    metadata = {"render.modes": []}

    def __init__(self, n_nodes=5, max_steps=None, reward_mode="dense", seed=None, safe_mode=True):
        super().__init__()
        self.n_nodes = n_nodes
        self.max_nodes = 12
        self.max_steps = max_steps if max_steps is not None else n_nodes * (n_nodes - 1)
        self.reward_mode = reward_mode
        self.safe_mode = safe_mode
        self.dataset = None
        self.sampler = None

        self.action_space = spaces.MultiDiscrete([n_nodes, n_nodes])
        self.observation_space = spaces.Dict({
            "d": spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_nodes,), dtype=np.float32),
            "pi": spaces.Box(low=-1, high=self.max_nodes - 1, shape=(self.max_nodes,), dtype=np.int32),
            "w": spaces.Box(low=-1.0, high=1.0, shape=(self.max_nodes, self.max_nodes), dtype=np.float32),
        })

        self.np_random, _ = gym.utils.seeding.np_random(seed)

    def _init_graph(self):
        """Initialise graph using CLRS dataset for Bellman-Ford, safe for CLRS."""
        import traceback
        import numpy as np
        from clrs import create_dataset

        try:
            # Create dataset iterator once
            if not hasattr(self, "dataset_iterator"):
                ds, spec, _ = create_dataset(
                    folder="./data/tmp/",
                    algorithm="bellman_ford",
                    batch_size=1,
                    split="train",
                )
                self.dataset_iterator = ds.as_numpy_iterator()

            feedback = next(self.dataset_iterator)

            # Extract adjacency matrix (A or adj)
            A = None
            for inp in feedback.features.inputs:
                if inp.name in ("A", "adj"):
                    A = np.array(inp.data[0])  # (1, N, N) → take [0]
                    break
            if A is None:
                raise ValueError("No adjacency matrix found in CLRS Bellman-Ford sample")

            A = np.array(A, dtype=np.float32)
            np.fill_diagonal(A, 0.0)
            self.w = A
            self.n_nodes = A.shape[0]

            # Extract predecessor pointers (pi)
            gt_pi = None
            for out in feedback.outputs:
                if out.name.lower() in ("pi", "predecessor", "predecessors", "parents"):
                    gt_pi = np.array(out.data[0], dtype=np.int32)
                    break

            # Ground truth distances are not provided by CLRS → compute manually
            self.gt_pi = gt_pi if gt_pi is not None else np.full(self.n_nodes, -1, dtype=np.int32)
            self._compute_ground_truth()

            # Pad to max_nodes
            if self.n_nodes < self.max_nodes:
                pad = self.max_nodes - self.n_nodes
                self.w = np.pad(self.w, ((0, pad), (0, pad)), constant_values=0.0)

        except Exception as e:
            print("Warning: CLRS sample failed, falling back to random graph.")
            print("Reason:", repr(e))
            traceback.print_exc()
            w = self.np_random.uniform(-1.0, 1.0, (self.n_nodes, self.n_nodes)).astype(np.float32)
            np.fill_diagonal(w, 0.0)
            if self.n_nodes < self.max_nodes:
                pad = self.max_nodes - self.n_nodes
                w = np.pad(w, ((0, pad), (0, pad)), constant_values=0.0)
            self.w = w
            self._compute_ground_truth()

        self.gt_d, self.gt_pi_ref = compute_ground_truth(self.w, s=0)

    def _get_obs(self):
        d = np.nan_to_num(self.d.copy(), nan=0.0, posinf=1e6, neginf=-1e6)
        pi = self.pi.copy()

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
                if self.reward_mode == "dense":
                    reward = 1.0  # reward for successful relaxation

        # Track whether any improvement happened in this pass
        self.updated_in_pass = self.updated_in_pass or improved

        # Count steps within the current relaxation pass
        self.steps_in_pass = getattr(self, "steps_in_pass", 0) + 1
        edges_per_pass = self.n_nodes * (self.n_nodes - 1)

        # After one full pass through all possible edges
        if self.steps_in_pass >= edges_per_pass:
            self.relax_passes += 1
            if not self.updated_in_pass:
                # No updates → converged early
                self.done_flag = True
                info_reason = "converged"
            elif self.relax_passes >= (self.n_nodes - 1):
                # Completed n-1 passes → stop
                self.done_flag = True
                info_reason = "max_passes"
            else:
                # Continue another pass
                info_reason = "next_pass"

            # Reset per-pass tracking
            self.steps_in_pass = 0
            self.updated_in_pass = False
        else:
            info_reason = "running"

        done = self.done_flag or self.steps >= self.max_steps
        obs = self._get_obs()
        info = {"reason": info_reason}

        return obs, reward, done, False, info

    def compute_metrics(self):
        gt_d, gt_pi, d = self.gt_d, self.gt_pi, self.d
        tol = 1e-4

        dist_match = np.zeros(self.n_nodes, dtype=bool)
        for i in range(self.n_nodes):
            if np.isinf(gt_d[i]) and np.isinf(d[i]):
                dist_match[i] = True
            elif np.isfinite(gt_d[i]) and np.isfinite(d[i]):
                dist_match[i] = np.isclose(gt_d[i], d[i], atol=tol)

        pointer_match = (self.pi == gt_pi)

        return {
            "distance_accuracy": float(dist_match.mean()),
            "pointer_accuracy": float(pointer_match.mean()),
            "distance_match_vector": dist_match,
            "pointer_match_vector": pointer_match,
        }
    
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
                if not np.isfinite(dist[u]):
                    continue  # skip unreachable nodes
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


    def render(self):
        print("\nEdge weights:\n", np.round(self.w[:self.n_nodes, :self.n_nodes], 3))
        print("Current distances:", np.round(self.d, 3))
        print("Predecessors:", self.pi)
        print("Ground truth distances:", np.round(self.gt_d, 3))
        print("Ground truth preds:", self.gt_pi)