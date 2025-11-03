from envs.bellman_ford_env import BellmanFordEnv
import numpy as np

def main():
    print("Creating environment...")
    env = BellmanFordEnv(n_nodes=5, reward_mode="dense", seed=42)
    obs, _ = env.reset()
    print("Initial observation:")
    env.render()

    total_reward = 0.0
    for step in range(10):
        u, v = env.action_space.sample()
        obs, reward, done, _, _ = env.step((u, v))
        total_reward += reward

        print(f"\n--- Step {step + 1} ---")
        print(f"Action: (u={u}, v={v})")
        print(f"Reward: {reward}")
        print("Observation (d):", np.round(obs["d"], 3))

        if done:
            print("\nEpisode finished.")
            break

    print("\nTotal reward collected:", total_reward)
    env.render()

    env.close()


if __name__ == "__main__":
    main()
