from envs.bellman_ford_env import BellmanFordEnv


def test_expert_action_follows_edge_list_scan_order():
    env = BellmanFordEnv(
        n_nodes=4,
        max_nodes=4,
        train_nodes=[4],
        fixed_nodes=4,
        reward_mode="sparse",
        seed=123,
        use_clrs=False,
    )

    obs, _ = env.reset(seed=123)
    assert len(env.edge_list) > 0

    for t in range(min(8, len(env.edge_list) * 2)):
        action = env.get_expert_action()
        u, v = env.action_to_edge(action)
        expected_u, expected_v = env.edge_list[env.t % len(env.edge_list)]
        assert (u, v) == (expected_u, expected_v)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break


def test_valid_action_mask_marks_exactly_edges_present():
    env = BellmanFordEnv(
        n_nodes=4,
        max_nodes=4,
        train_nodes=[4],
        fixed_nodes=4,
        reward_mode="sparse",
        seed=456,
        use_clrs=False,
    )

    env.reset(seed=456)
    mask = env.get_valid_action_mask()

    assert mask.shape == (env.max_nodes * env.max_nodes,)
    assert mask.sum() == len(env.edge_list)

    for u, v in env.edge_list:
        action = env.edge_to_action(u, v)
        assert mask[action] == 1.0