import os
import numpy as np
from stable_baselines3 import PPO
from envs.bellman_ford_env import BellmanFordEnv

def main():
    checkpoint_path = "./checkpoints/bf_agent_final.zip"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError("Checkpoint not found. Train the agent first!")

    env = BellmanFordEnv(n_nodes=5, reward_mode="dense")
    model = PPO.load(checkpoint_path, env=env)

    obs, _ = env.reset()
    total_reward = 0.0

    print("Evaluating trained Bellman-Ford agent...\n")

    for step in range(50):
        action, _ = model.predict(obs, deterministic=True)
        if isinstance(action, (list, tuple, np.ndarray)):
            action = int(np.asarray(action).item())
        obs, reward, done, _, info = env.step(action)
        total_reward += reward
        u, v = env.edges[action]

        print(f"--- Step {step + 1} ---")
        print(f"Action chosen: index={action} -> (u={u}, v={v})")
        print(f"Reward: {reward}")
        print(f"Distances: {np.round(obs['d'], 3)}")
        print("Info:", info)
        print()

        if done:
            print("Episode finished early. Reason:", info.get("reason"))
            break

    print(f"Total reward collected: {total_reward:.2f}")

    metrics = env.compute_metrics()
    print("Final metrics:", metrics)
    env.close()


if __name__ == "__main__":
    main()
