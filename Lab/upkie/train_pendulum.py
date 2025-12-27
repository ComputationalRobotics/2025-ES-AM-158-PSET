import upkie.envs
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import os

upkie.envs.register()

# Configure the environment
ENV_ID = "Upkie-Spine-Pendulum"
ENV_KWARGS = dict(frequency=200.0, regulate_frequency=False)
TOTAL_TIMESTEPS = 500_000
N_ENVS = 4
SEED = 67

# PPO hyperparameters
PPO_PARAMS = {
    "policy": "MlpPolicy",
    "learning_rate": 0.0003,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.0,
    "policy_kwargs": dict(net_arch=[64, 64]),
    "verbose": 1,
}

def main():
    print("=" * 70)
    print("Task 2: Training Upkie-Pendulum Full Body Stabilization")
    print("=" * 70)

    os.makedirs("./models", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)

    # Create vectorized training environment
    train_env = make_vec_env(ENV_ID, n_envs=N_ENVS, env_kwargs=ENV_KWARGS, seed=SEED)

    # Create separate evaluation environment
    eval_env = make_vec_env(ENV_ID, n_envs=1, env_kwargs=ENV_KWARGS, seed=SEED + 1)

    # Callbacks for saving best model and checkpoints
    eval_callback = EvalCallback(eval_env, best_model_save_path="./models", log_path="./logs", eval_freq=10000, deterministic=True, render=False, n_eval_episodes=5)

    checkpoint_callback = CheckpointCallback(save_freq=50000, save_path="./models/checkpoints", name_prefix="ppo_upkie")

    # Create PPO agent
    model = PPO(env=train_env, tensorboard_log="./logs/tensorboard", device="auto", **PPO_PARAMS)

    print(f"Starting training for {TOTAL_TIMESTEPS} total timesteps")
    print(f"Training with {N_ENVS} environments")
    print(f"Model architecture: {PPO_PARAMS['policy_kwargs']}")

    # Train teh agent
    model.learn(TOTAL_TIMESTEPS, callback=[eval_callback, checkpoint_callback], progress_bar=True)

    model.save("./models/ppo_upkie_final")
    print("Training complete. Model saved to ./models/ppo_upkie_final")
    print("Best model saved to ./models/best_model.zip")

if __name__ == "__main__":
    main()