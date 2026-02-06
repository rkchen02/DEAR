import os
import warnings
from typing import Any, Dict, Optional, Tuple
import random

import gymnasium as gym
from gymnasium import spaces
import numpy as np


class BellmanFordEnv(gym.Env):
    """
    Bellman–Ford environment for RL.

    Optionally draws graphs from the CLRS / DEAR generators
    (via the PyTorch Geometric dataset defined in datasets/clrs_datasets.py).

    Key points:
    - State is the current distance estimates d, predecessors pi and a visited mask.
    - Action is choosing a directed edge (u, v) to relax.
    - Transition follows the usual Bellman–Ford relaxation rule.
    - Episode length follows the classical CLRS Bellman–Ford structure:
        horizon = (|V| - 1) * |E|
      i.e. we perform up to |V| - 1 passes over all edges; no early stopping
      when there are no more improvements.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        n_nodes: int,
        reward_mode: str = "dense",
        seed: Optional[int] = None,
        use_clrs: bool = True,
        clrs_root: Optional[str] = None,
        clrs_num_samples: int = 10_000,
        clrs_split: str = "train",
        clrs_sampler_type: str = "normal",
        clrs_randomise_pos: bool = True,
        max_nodes: Optional[int] = None,
        train_nodes: Optional[list[int]] = None,
        fixed_nodes: Optional[int] = None,
        clrs_num_nodes_list: Optional[list[int]] = None,
        allow_negative_weights: bool = True,
        min_weight: float = -1.0,
        max_weight: float = 1.0,
    ) -> None:
        """
        Parameters
        ----------
        n_nodes : int
            Number of nodes in the graph. For CLRS graphs this must match the
            `num_nodes` argument used when creating the CLRS dataset.
        reward_mode : {"dense", "sparse"}
            Dense: +1 for every successful relaxation, 0 otherwise.
            Sparse: reward only when episode terminates, with +1 if success else 0.
        seed : int, optional
            Random seed.
        use_clrs : bool
            If True, try to use the CLRS / DEAR graph generator
            (datasets.clrs_datasets.CLRS with algorithm="bellman_ford").
            If this import / construction fails, fall back to random graphs.
        clrs_root : str, optional
            Root directory for CLRS data. If None, defaults to
            "<repo_root>/data/clrs".
        clrs_num_samples : int
            Number of CLRS samples to pre-generate in the underlying dataset.
        clrs_split : str
            Split to use for CLRS ("train", "val", "test", ...).
        clrs_sampler_type : str
            Sampler type passed through to CLRS (e.g. "normal").
        clrs_randomise_pos : bool
            Whether to randomise node positions in CLRS graphs.
        allow_negative_weights : bool
            Whether edges may have negative weights (no negative cycles).
        min_weight, max_weight : float
            Range for random weights when not using CLRS.
        """
        super().__init__()

        assert reward_mode in ("dense", "sparse")
        self.n_nodes = int(n_nodes)
        self.max_nodes = int(max_nodes) if max_nodes is not None else int(n_nodes)
        if self.max_nodes < self.n_nodes:
            raise ValueError(f"max_nodes ({self.max_nodes}) must be >= n_nodes ({self.n_nodes})")

        if train_nodes is None:
            train_nodes = [int(n_nodes)]
        self.train_nodes = [int(x) for x in train_nodes]
        self.fixed_nodes = int(fixed_nodes) if fixed_nodes is not None else None

        self.current_n_nodes = int(n_nodes)
        self.reward_mode = reward_mode
        self.rng = np.random.RandomState(seed)
        self.allow_negative_weights = allow_negative_weights
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.max_steps = max(1, (self.max_nodes - 1) * self.max_nodes)

        # --- CLRS integration -------------------------------------------------
        self.use_clrs = use_clrs
        self._clrs_datasets: Dict[int, Any] = {}
        self._clrs_idx: Dict[int, int] = {}

        if self.use_clrs:
            try:
                # Lazy import so that this file can still be used without CLRS.
                from datasets.clrs_datasets import CLRS as CLRSData  # type: ignore

                if clrs_root is None:
                    # Assume this file lives at <repo_root>/envs/bellman_ford_env.py
                    repo_root = os.path.abspath(
                        os.path.join(os.path.dirname(__file__), os.pardir)
                    )
                    clrs_root = os.path.join(repo_root, "data", "clrs")

                if clrs_num_nodes_list is None:
                    sizes = set(self.train_nodes)
                    if self.fixed_nodes is not None:
                        sizes.add(self.fixed_nodes)
                    sizes.add(self.n_nodes)
                    clrs_num_nodes_list = sorted(sizes)
                else:
                    clrs_num_nodes_list = [int(x) for x in clrs_num_nodes_list]

                for nn in clrs_num_nodes_list:
                    ds = CLRSData(
                        root=clrs_root,
                        num_nodes=nn,
                        num_samples=clrs_num_samples,
                        algorithm="bellman_ford",
                        split=clrs_split,
                        sampler_type=clrs_sampler_type,
                        randomise_pos=clrs_randomise_pos,
                        seed=seed if seed is not None else 0,
                    )
                    if len(ds) == 0:
                        warnings.warn(
                            f"CLRS dataset bellman_ford for num_nodes={nn} is empty; skipping."
                        )
                        continue
                    self._clrs_datasets[nn] = ds
                    self._clrs_idx[nn] = 0

                if not self._clrs_datasets:
                    warnings.warn(
                        "CLRS dataset for bellman_ford is empty; "
                        "falling back to random graphs."
                    )
                    self._clrs_datasets = {}
                    self.use_clrs = False
            except Exception as e:
                warnings.warn(
                    f"Could not construct CLRS bellman_ford dataset "
                    f"(error: {e!r}); falling back to random graphs.\n"
                    "For CLRS graphs, make sure that "
                    "datasets/clrs_datasets.py is on the Python path and that "
                    "prepare_datasets.py has been run for bellman_ford."
                )
                self.use_clrs = False

        # --- Gym spaces -------------------------------------------------------
        # Action: choose a directed edge (u, v).
        self.action_space = spaces.Discrete(self.max_nodes * self.max_nodes)

        # Observation: dict of numpy arrays.
        self.observation_space = spaces.Dict(
            {
                "t": spaces.Box(0, np.iinfo(np.int32).max, shape=(1,), dtype=np.int32),
                "p": spaces.Box(1, 2, shape=(1,), dtype=np.int32),
                "source": spaces.Discrete(self.max_nodes),
                "d": spaces.Box(-np.inf, np.inf, shape=(self.max_nodes,), dtype=np.float32),
                "visited": spaces.Box(0.0, 1.0, shape=(self.max_nodes,), dtype=np.float32),
                "pred": spaces.Box(-1, self.max_nodes - 1, shape=(self.max_nodes,), dtype=np.int32),
            }
        )


        # Internal state placeholders (initialised in reset()).
        self.W: Optional[np.ndarray] = None  # weight matrix [n, n], inf if no edge
        self.edge_list = []  # list of (u, v) with an edge
        self.source: int = 0
        self.d: np.ndarray = np.full(self.max_nodes, np.inf, dtype=np.float32)
        self.pred: np.ndarray = -np.ones(self.max_nodes, dtype=np.int32)
        self.visited: np.ndarray = np.zeros(self.max_nodes, dtype=np.float32)
        self.t: int = 0
        self.phase: int = 1
        self.max_steps: int = 0  # set after we know |E|
        self.opt_d: Optional[np.ndarray] = None
        self.opt_pred: Optional[np.ndarray] = None
        self.has_neg_cycle: bool = False

    # ------------------------------------------------------------------ utils
    def _empty_obs(self) -> Dict[str, Any]:
        return {
            "t": np.array([0], dtype=np.int32),
            "p": np.array([1], dtype=np.int32),
            "source": 0,
            "d": np.full((self.max_nodes,), np.inf, dtype=np.float32),
            "visited": np.zeros((self.max_nodes,), dtype=np.float32),
            "pred": np.full((self.max_nodes,), -1, dtype=np.int32),
        }

    def _graph_from_clrs_sample(self, num_nodes: int) -> Tuple[np.ndarray, int]:
        """Get a graph from the CLRS dataset and convert it to a padded (W, n)."""
        ds = self._clrs_datasets.get(num_nodes)
        if ds is None:
            raise RuntimeError(
                f"Requested CLRS num_nodes={num_nodes} but no dataset was loaded for that size. "
                f"Loaded sizes: {sorted(self._clrs_datasets.keys())}"
            )
        idx = int(self._clrs_idx.get(num_nodes, 0))
        self._clrs_idx[num_nodes] = (idx + 1) % len(ds)

        data = ds[idx]

        data_num_nodes = int(data.num_nodes)
        n = min(data_num_nodes, self.max_nodes)

        # CLRS graph representation (from datasets/clrs_datasets.py):
        # - data.edge_index : LongTensor [2, E]
        # - data.A          : FloatTensor [E] or [E, 1] with edge weights
        edge_index = data.edge_index.cpu().numpy()
        edge_attr = data.A.view(-1).cpu().numpy().astype(np.float32)

        # Padded adjacency/weight matrix of fixed size (max_nodes x max_nodes).
        W = np.full((self.max_nodes, self.max_nodes), np.inf, dtype=np.float32)

        # Fill only edges that lie inside the first n nodes.
        for (u, v), w in zip(edge_index.T, edge_attr):
            u = int(u)
            v = int(v)
            if u < n and v < n:
                W[u, v] = float(w)

        return W, n


    def _graph_random(self) -> Tuple[np.ndarray, int]:
        """Fallback random graph generator (Erdos–Rényi style)."""
        n = int(self.n_nodes)
        W = np.full((self.max_nodes, self.max_nodes), np.inf, dtype=np.float32)

        p_edge = 0.5
        for u in range(n):
            for v in range(n):
                if u == v:
                    continue
                if self.rng.rand() < p_edge:
                    if self.allow_negative_weights:
                        w = self.rng.uniform(self.min_weight, self.max_weight)
                    else:
                        w = self.rng.uniform(0.0, self.max_weight)
                    W[u, v] = float(w)

        # Ensure at least some edges.
        if not np.any(np.isfinite(W)):
            for u in range(n - 1):
                W[u, u + 1] = self.rng.uniform(
                    0.0 if not self.allow_negative_weights else self.min_weight,
                    self.max_weight,
                )

        return W, n

    def _build_edge_list(self) -> None:
        assert self.W is not None
        self.edge_list = [
            (int(u), int(v))
            for u in range(int(self.current_n_nodes))
            for v in range(int(self.current_n_nodes))
            if np.isfinite(self.W[u, v])
        ]

    def _bellman_ford_ground_truth(
        self, source: int
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """Classical Bellman–Ford run used for targets / metrics."""
        n = int(self.current_n_nodes)
        d = np.full(n, np.inf, dtype=np.float32)
        pred = -np.ones(n, dtype=np.int32)
        d[source] = 0.0

        # |V| - 1 relaxation passes.
        for _ in range(n - 1):
            for u, v in self.edge_list:
                w = self.W[u, v]
                if not np.isfinite(w):
                    continue
                if not np.isfinite(d[u]):
                    continue
                new_d = d[u] + w
                if new_d < d[v]:
                    d[v] = new_d
                    pred[v] = u

        # Negative cycle detection.
        has_neg_cycle = False
        for u, v in self.edge_list:
            w = self.W[u, v]
            if not np.isfinite(w):
                continue
            if not np.isfinite(d[u]):
                continue
            if d[u] + w < d[v]:
                has_neg_cycle = True
                break

        return d, pred, has_neg_cycle
    
    def _pad_1d(self, arr, *, fill, dtype):
        """Pad/truncate a 1D array-like to length self.max_nodes."""
        a = np.asarray(arr, dtype=dtype)
        if a.shape == (self.max_nodes,):
            return a
        out = np.full((self.max_nodes,), fill, dtype=dtype)
        n = min(a.shape[0], self.max_nodes)
        out[:n] = a[:n]
        return out

    def _get_obs(self) -> Dict[str, Any]:
        # Always return fixed-size tensors that match observation_space.
        d = self._pad_1d(self.d, fill=np.inf, dtype=np.float32)
        D_INF = np.float32(1e6)
        d = np.where(np.isinf(d), D_INF, d).astype(np.float32)
        pred = self._pad_1d(self.pred, fill=-1, dtype=np.int32)
        visited = self._pad_1d(self.visited, fill=0.0, dtype=np.float32)
        return {
            "d": d,
            "pred": pred,
            "visited": visited,
            "source": int(self.source),
            "t": np.array([self.t], dtype=np.int32),
            "p": np.array([self.phase], dtype=np.int32),
        }

    # ----------------------------------------------------------------- Env API
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ):
        super().reset(seed=seed)
        if seed is not None:
            self.rng.seed(seed)

        # Sample graph (CLRS if available, else random).
        if self.use_clrs and self._clrs_datasets:
            chosen_n = int(self.fixed_nodes) if self.fixed_nodes is not None else int(random.choice(self.train_nodes))
            self.W, self.current_n_nodes = self._graph_from_clrs_sample(chosen_n)
        else:
            self.W, self.current_n_nodes = self._graph_random()

        # Build edge list & classical BF ground truth.
        self._build_edge_list()
        num_edges = len(self.edge_list)
        if num_edges == 0:
            raise RuntimeError("Graph has no edges; this should not happen.")

        # Horizon matches classical CLRS Bellman–Ford: (|V| - 1) * |E|.
        self.max_steps = max(1, (self.current_n_nodes - 1) * num_edges)

        # Sample a source node.
        self.source = int(self.rng.randint(self.current_n_nodes))

        # Initialise distances / predecessors.
        self.d = np.full(self.max_nodes, np.inf, dtype=np.float32)
        self.pred = np.full(self.max_nodes, -1, dtype=np.int32)
        self.visited = np.zeros(self.max_nodes, dtype=np.float32)
        self.d[self.source] = 0.0
        self.visited[:] = 0.0
        self.visited[self.source] = 1.0

        self.t = 0
        self.phase = 1

        # Ground-truth Bellman–Ford solution for metrics / shaping.
        self.opt_d, self.opt_pred, self.has_neg_cycle = self._bellman_ford_ground_truth(
            self.source
        )

        obs = self._get_obs()
        info: Dict[str, Any] = {}
        return obs, info

    def step(self, action):
        """
        Perform one Bellman–Ford relaxation step on edge (u, v) chosen by the agent.
        """
        if isinstance(action, (np.integer, int)):
            a = int(action)
            u = a // self.max_nodes
            v = a % self.max_nodes
        elif isinstance(action, (list, tuple, np.ndarray)) and len(action) == 2:
            u, v = int(action[0]), int(action[1])
        else:
            raise ValueError(f"Unexpected action format: {action!r}")
        
        if u < 0 or v < 0 or u >= self.current_n_nodes or v >= self.current_n_nodes:
            # Penalise/ignore invalid actions; keep episode going.
            obs = self._get_obs()
            info: Dict[str, Any] = {"t": int(self.t), "relaxed": False, "is_success": 0.0}
            return obs, -1.0, False, False, info

        reward = 0.0
        relaxed = False

        w = self.W[u, v]
        if np.isfinite(w) and np.isfinite(self.d[u]):
            new_d = self.d[u] + w
            if new_d < self.d[v]:
                relaxed = True
                self.d[v] = new_d
                self.pred[v] = u
                self.visited[v] = 1.0
                if self.reward_mode == "dense":
                    reward = 1.0
        # else: no-op; dense gives 0, sparse gives 0 (until terminal bonus)

        self.t += 1

        terminated = False
        truncated = False

        if self.t >= self.max_steps:
            terminated = True

        info: Dict[str, Any] = {
            "t": int(self.t),
            "relaxed": bool(relaxed),
        }

        if terminated:
            solved = bool(self._is_solved())
            info["is_success"] = solved
            if self.reward_mode == "sparse" and solved:
                reward += 1.0
        
        if terminated or truncated:
            metrics = self.compute_metrics()
            info.update(metrics)

        obs = self._get_obs()
        return obs, float(reward), terminated, truncated, info

    def _is_solved(self) -> bool:
        """Return True if the current (d, pred) match the reference solution."""
        if self.opt_d is None or self.opt_pred is None:
            return False

        n = int(self.current_n_nodes)

        d = np.asarray(self.d[:n], dtype=np.float32)
        opt = np.asarray(self.opt_d[:n], dtype=np.float32)

        finite_mask = np.isfinite(opt) & np.isfinite(d)
        inf_both = (~np.isfinite(opt)) & (~np.isfinite(d))

        if not np.allclose(d[finite_mask], opt[finite_mask], atol=1e-5, rtol=0.0):
            return False
        if not np.all(inf_both | finite_mask):
            return False

        reachable = np.isfinite(opt)
        pred = np.asarray(self.pred[:n], dtype=np.int32)
        opt_pred = np.asarray(self.opt_pred[:n], dtype=np.int32)

        src = int(self.source)
        ok = (pred == opt_pred) & reachable
        if src < n and reachable[src]:
            ok[src] = True

        return bool(np.all(ok[reachable]))

    # ------------------------------------------------------------- diagnostics
    def compute_metrics(self) -> Dict[str, float]:
        """
        Compare current distances / preds to the classical Bellman–Ford solution.
        """
        if self.opt_d is None or self.opt_pred is None:
            return {}
        
        n = int(self.current_n_nodes)
        d = np.asarray(self.d[:n], dtype=np.float32)
        opt_d = np.asarray(self.opt_d[:n], dtype=np.float32)

        reachable = np.isfinite(opt_d)
        finite_both = reachable & np.isfinite(d)
        mismatch_reachability = reachable ^ np.isfinite(d)

        mae = float(np.mean(np.abs(d[finite_both] - opt_d[finite_both]))) if np.any(finite_both) else 0.0
        penalty = float(np.mean(mismatch_reachability.astype(np.float32))) * 1e3
        dist_error = mae + penalty

        pred = np.asarray(self.pred[:n], dtype=np.int32)
        opt_pred = np.asarray(self.opt_pred[:n], dtype=np.int32)
        pred_accuracy = float(np.mean((pred[reachable] == opt_pred[reachable]).astype(np.float32))) if np.any(reachable) else 0.0

        return {
            "dist_error": dist_error,
            "pred_accuracy": pred_accuracy,
            "has_neg_cycle": float(self.has_neg_cycle),
        }

    def render(self):
        """Pretty-print the current state (for debugging)."""
        print(f"t = {self.t}, phase p = {self.phase}")
        print(f"source = {self.source}")
        print("Distances d:", self.d)
        print("Visited mask:", self.visited)
        print("Predecessors:", self.pred)