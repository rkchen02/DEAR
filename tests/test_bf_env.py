from envs.bellman_ford_env import BellmanFordEnv

def main():
    env = BellmanFordEnv(n_nodes=16, use_clrs=True)
    obs, info = env.reset()

    print("use_clrs:", env.use_clrs)
    print("n_nodes:", env.n_nodes)
    print("W shape:", env.W.shape)
    print("num_edges:", len(env.edge_list))
    print("max_steps (horizon):", env.max_steps)

    env.render()

    for _ in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"t={obs['t']}, action={action}, reward={reward}")

    metrics = env.compute_metrics()
    print(metrics)

if __name__ == "__main__":
    main()
