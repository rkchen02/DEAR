import os
import csv
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure

from envs.bellman_ford_env import BellmanFordEnv


def evaluate_model(model, n_nodes_list, n_eval_episodes=10):
    """Evaluate a trained PPO model on different graph sizes."""
    all_results = []

    for n in n_nodes_list:
        print(f"\n Evaluating on graphs with {n} nodes...")
        env = BellmanFordEnv(n_nodes=n, reward_mode="dense")
        distance_accs, pointer_accs, rewards = [], [], []

        for ep in range(n_eval_episodes):
            obs, _ = env.reset()
            done = False
            total_reward = 0.0

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _, info = env.step(action)
                total_reward += reward

            # Compute CLRS-style metrics from the environment
            metrics = env.compute_metrics()
            distance_accs.append(metrics["distance_accuracy"])
            pointer_accs.append(metrics["pointer_accuracy"])
            rewards.append(total_reward)

            print(
                f"Episode {ep+1}: "
                f"Reward={total_reward:.1f}, "
                f"DistAcc={metrics['distance_accuracy']:.2f}, "
                f"PtrAcc={metrics['pointer_accuracy']:.2f}"
            )

        env.close()

        mean_dist_acc = np.mean(distance_accs)
        mean_ptr_acc = np.mean(pointer_accs)
        mean_reward = np.mean(rewards)

        print(f" Mean results for n={n}: "
              f"Reward={mean_reward:.2f}, "
              f"DistAcc={mean_dist_acc:.2f}, "
              f"PtrAcc={mean_ptr_acc:.2f}")

        all_results.append({
            "n_nodes": n,
            "mean_reward": mean_reward,
            "distance_accuracy": mean_dist_acc,
            "pointer_accuracy": mean_ptr_acc,
        })

    return all_results


def main():
    log_dir = "./logs/train_eval"
    ckpt_dir = "./checkpoints"
    results_csv = os.path.join(log_dir, "train_eval_results.csv")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    print("\n Training PPO agent on small graphs (n=4–6)...")
    env = BellmanFordEnv(n_nodes=5, reward_mode="dense")
    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
    )

    new_logger = configure(log_dir, ["stdout", "tensorboard"])
    model.set_logger(new_logger)

    ckpt_callback = CheckpointCallback(save_freq=10_000, save_path=ckpt_dir, name_prefix="bf_runner")

    total_timesteps = 200_000
    model.learn(total_timesteps=total_timesteps, callback=ckpt_callback)
    model.save(os.path.join(ckpt_dir, "bf_runner_final.zip"))
    env.close()

    print("\n Training complete.")

    print("\n Evaluating trained agent...")
    eval_model = PPO.load(os.path.join(ckpt_dir, "bf_runner_final.zip"))

    small_graphs = [4, 5, 6]
    large_graphs = [8, 10, 12]

    print("\n Evaluating on SMALL graphs (train regime)...")
    small_results = evaluate_model(eval_model, small_graphs, n_eval_episodes=10)

    print("\n Evaluating on LARGE graphs (generalization regime)...")
    large_results = evaluate_model(eval_model, large_graphs, n_eval_episodes=10)

    with open(results_csv, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["n_nodes", "mean_reward", "distance_accuracy", "pointer_accuracy"])
        writer.writeheader()
        writer.writerows(small_results + large_results)

    print(f"\n Results saved to {results_csv}")
    print("\n Experiment complete.")


if __name__ == "__main__":
    main()
