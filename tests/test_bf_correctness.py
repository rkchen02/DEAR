import numpy as np

from envs.bellman_ford_env import BellmanFordEnv

def classical_bellman_ford_numpy(W: np.ndarray, source: int, eps: float = 1e-6):
    """
    Pure-NumPy Bellman–Ford on a dense adjacency matrix W with
    W[u, v] = weight or +inf if no edge.

    Returns:
        d      : np.ndarray of shape [n], shortest distances
        pred   : np.ndarray of shape [n], predecessors (-1 if none)
        has_nc : bool, whether a negative cycle is detected
    """
    n = W.shape[0]
    d = np.full(n, np.inf, dtype=np.float64)
    pred = np.full(n, -1, dtype=np.int64)

    d[source] = 0.0

    # Build edge list (u, v) where W[u, v] is finite
    edges = [(int(u), int(v))
             for u in range(n)
             for v in range(n)
             if np.isfinite(W[u, v])]

    # n - 1 relaxation passes
    for _ in range(n - 1):
        updated = False
        for (u, v) in edges:
            w = W[u, v]
            if not np.isfinite(d[u]) or not np.isfinite(w):
                continue
            new_d = d[u] + w
            if new_d + eps < d[v]:
                d[v] = new_d
                pred[v] = u
                updated = True
        if not updated:
            break

    # One more pass to detect negative cycles
    has_neg_cycle = False
    for (u, v) in edges:
        w = W[u, v]
        if not np.isfinite(d[u]) or not np.isfinite(w):
            continue
        if d[u] + w + eps < d[v]:
            has_neg_cycle = True
            break

    return d, pred, has_neg_cycle


def make_env_for_correctness(n_nodes: int, seed: int) -> BellmanFordEnv:
    """
    We:
      - disable CLRS graphs (use_clrs=False) to avoid any JAX / TF baggage
      - disable negative weights so there are no negative cycles
    """
    env = BellmanFordEnv(
        n_nodes=n_nodes,
        seed=seed,
        use_clrs=False,
        allow_negative_weights=False,
        min_weight=0.0,
        max_weight=5.0,
    )
    return env


def run_expert_policy(env: BellmanFordEnv):
    """
    Reset the env and then apply the classical Bellman–Ford relaxation
    schedule:
        for pass in range(n_nodes - 1):
            for (u, v) in edges:
                relax edge (u, v)
    using env.step([...]) as the relaxation primitive.
    """
    obs, info = env.reset()

    n = env.n_nodes
    # env.edge_list is already a list of (u, v) pairs where W[u, v] is finite
    edges = list(env.edge_list)

    for _ in range(n - 1):
        for (u, v) in edges:
            action = np.array([u, v], dtype=np.int64)
            obs, reward, terminated, truncated, info = env.step(action)
            # We don't *expect* termination early (horizon is quite large),
            # but if someone ever adds a smarter termination, this keeps us safe.
            if terminated or truncated:
                return obs

    return obs


def assert_same_solution(W: np.ndarray, source: int, obs, eps: float = 1e-5):
    """
    Compare the env's final distances / predecessors with the classical
    Bellman–Ford solution.
    """
    d_star, pred_star, has_neg_cycle = classical_bellman_ford_numpy(W, source)

    # By construction (allow_negative_weights=False) there should be no neg cycles
    assert not has_neg_cycle, "Test graph unexpectedly has a negative cycle."

    d_env = np.asarray(obs["d"], dtype=np.float64)
    pred_env = np.asarray(obs["pred"], dtype=np.int64)

    assert d_env.shape == d_star.shape, "Distance vector shape mismatch."
    assert pred_env.shape == pred_star.shape, "Predecessor vector shape mismatch."

    n = d_star.shape[0]

    for i in range(n):
        # Distances: handle infinities carefully
        if not np.isfinite(d_star[i]) and not np.isfinite(d_env[i]):
            # Both +inf -> unreachable, that's fine
            continue

        # If classical says reachable, env must also say reachable
        assert np.isfinite(d_star[i]), f"Ground-truth d[{i}] is inf unexpectedly."
        assert np.isfinite(d_env[i]), f"Env d[{i}] is inf but GT is {d_star[i]}."

        diff = abs(float(d_env[i]) - float(d_star[i]))
        assert diff <= eps, (
            f"Distance mismatch at node {i}: "
            f"env={d_env[i]}, gt={d_star[i]}, |diff|={diff}"
        )

        # Predecessors: only meaningful for reachable non-source nodes
        if i == source:
            continue
        if not np.isfinite(d_star[i]):
            continue

        assert int(pred_env[i]) == int(pred_star[i]), (
            f"Predecessor mismatch at node {i}: "
            f"env={pred_env[i]}, gt={pred_star[i]}"
        )


def test_bellman_ford_matches_classical():
    rng = np.random.default_rng(0)

    num_tests = 5
    for idx in range(num_tests):
        n_nodes = int(rng.integers(4, 12))
        seed = int(rng.integers(0, 10_000))

        env = make_env_for_correctness(n_nodes=n_nodes, seed=seed)
        obs_final = run_expert_policy(env)

        # Compare env result vs classical Bellman–Ford
        assert_same_solution(env.W, env.source, obs_final)

        print(
            f"[OK] test graph {idx}: "
            f"n_nodes={n_nodes}, seed={seed}, source={env.source}"
        )

    print(f"All {num_tests} Bellman–Ford correctness tests passed!")


def main():
    test_bellman_ford_matches_classical()


if __name__ == "__main__":
    main()
