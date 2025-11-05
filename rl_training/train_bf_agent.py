import os
import time
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.logger import configure

from envs.bellman_ford_env import BellmanFordEnv


class ProgressCallback(BaseCallback):
    """
    Custom callback for printing live progress updates every `print_freq` steps.
    """
    def __init__(self, total_timesteps, print_freq=5000, verbose=1):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.print_freq = print_freq
        self.start_time = None

    def _on_training_start(self):
        self.start_time = time.time()
        print(f"\nTraining started: {self.total_timesteps:,} total timesteps\n")

    def _on_step(self):
        # Print every N steps
        if self.num_timesteps % self.print_freq == 0:
            elapsed = time.time() - self.start_time
            progress = self.num_timesteps / self.total_timesteps
            eta = (elapsed / progress) - elapsed if progress > 0 else 0
            print(f"Progress: {progress*100:5.1f}% | "
                  f"Steps: {self.num_timesteps:,}/{self.total_timesteps:,} | "
                  f"Elapsed: {elapsed/60:5.1f}m | ETA: {eta/60:5.1f}m")
        return True

    def _on_training_end(self):
        total_time = (time.time() - self.start_time) / 60
        print(f"\nTraining complete in {total_time:.1f} minutes!\n")

def make_env(seed=42):
    """Create Bellman-Ford environment."""
    return BellmanFordEnv(n_nodes=5, reward_mode="dense", seed=seed)


def main():
    log_dir = "./logs/bf_train"
    checkpoint_dir = "./checkpoints"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    env = make_env()

    # Configure logger for TensorBoard
    new_logger = configure(log_dir, ["stdout", "tensorboard"])

    # Instantiate PPO agent
    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
    )
    model.set_logger(new_logger)

    # Save checkpoints every 10 000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=checkpoint_dir,
        name_prefix="bf_agent"
    )

    # Train for 100 000 timesteps
    progress_callback = ProgressCallback(total_timesteps=100_000, print_freq=5000)

    model.learn(total_timesteps=100_000, callback=checkpoint_callback)

    # Save final model
    model.save(os.path.join(checkpoint_dir, "bf_agent_final"))
    env.close()


if __name__ == "__main__":
    main()
