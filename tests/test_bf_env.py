from envs.bellman_ford_env import BellmanFordEnv
import numpy as np

def main():
    print("Creating environment...")
    env = BellmanFordEnv(n_nodes=5, reward_mode="dense", seed=42, max_steps=500)
    obs, _ = env.reset()
    print("Initial observation:")
    env.render()

    total_reward = 0.0
    for step in range(50):
        action = int(env.action_space.sample())
        u, v = env.edges[action]
        obs, reward, done, _, info = env.step(action)
        total_reward += reward

        print(f"\n--- Step {step + 1} ---")
        print(f"Action index: {action} -> (u={u}, v={v})")
        print(f"Reward: {reward}")
        print("Observation (d):", np.round(obs["d"], 3))
        print("Info:", info)

        if done:
            print("\nEpisode finished. Reason:", info.get("reason"))
            break

    print("\nTotal reward collected:", total_reward)
    env.render()

    metrics = env.compute_metrics()
    print("\nMetrics:", {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()})

    env.close()


if __name__ == "__main__":
    main()
