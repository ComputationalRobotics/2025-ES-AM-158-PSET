#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import torch
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import TimeLimit

import upkie.envs
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback

# Import wrappers from rollout_policy_servos.py
from rollout_policy_servos import ServoVelActionWrapper, ServoObsFlattenWrapper
from servos_reward_wrapper import ServosRewardWrapper

# Register Upkie environments
upkie.envs.register()

# =====================================================================
# Configuration
# =====================================================================

TOTAL_TIMESTEPS = 1_000_000  # More timesteps needed for complex task
N_ENVS = 1  # Spine backend only reliable with single environment
SEED = 42
FREQUENCY_HZ = 200.0
MAX_STEPS = 300

# PPO Hyperparameters (tuned for servos task)
PPO_PARAMS = {
    "policy": "MlpPolicy",
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,  # Slight exploration bonus for complex task
    "policy_kwargs": dict(
        net_arch=[128, 128],  # Bigger network for more complex task
        activation_fn=torch.nn.ReLU
    ),
    "verbose": 1,
}

# Controller gains (same as in rollout_policy_servos.py)
GAINS = {
    "kp_wheel": 0.0,   # Wheels: velocity control only
    "kd_wheel": 1.7,
    "kp_leg": 2.0,     # Legs: position control
    "kd_leg": 1.7,
}

# =====================================================================
# Environment Wrapper Function (for make_vec_env)
# =====================================================================

def wrap_env(env):
    """
    Apply wrappers to the Upkie-Servos environment.
    
    Args:
        env: Base Upkie-Servos environment
        
    Returns:
        Wrapped environment ready for training
    """
    # Apply action wrapper: Dict action -> Box action
    env = ServoVelActionWrapper(env, fixed_order=None, gains=GAINS)
    
    # Apply reward wrapper: Add reward shaping and termination conditions
    env = ServosRewardWrapper(env, fall_pitch=1.0, max_ground_position=5.0)
    
    # Apply observation wrapper: Dict obs -> flat Box obs
    env = ServoObsFlattenWrapper(env)
    
    # Apply time limit
    env = TimeLimit(env, max_episode_steps=MAX_STEPS)
    
    return env

# =====================================================================
# Main Training Function
# =====================================================================

def main():
    print("=" * 70)
    print("Task 3: Training Upkie-Servos Full Body Stabilization")
    print("=" * 70)
    
    # Create save directories
    os.makedirs("./models_servos", exist_ok=True)
    os.makedirs("./logs_servos", exist_ok=True)
    os.makedirs("./models_servos/checkpoints", exist_ok=True)
    
    print(f"\nConfiguration:")
    print(f"  Total timesteps: {TOTAL_TIMESTEPS:,}")
    print(f"  Parallel environments: {N_ENVS}")
    print(f"  Max episode steps: {MAX_STEPS}")
    print(f"  Network architecture: {PPO_PARAMS['policy_kwargs']['net_arch']}")
    print(f"  Entropy coefficient: {PPO_PARAMS['ent_coef']}")
    
    # Environment kwargs
    env_kwargs = {
        "frequency": FREQUENCY_HZ,
        "regulate_frequency": False,  # Don't regulate during training
    }
    
    # Create vectorized training environment using make_vec_env (like Task 2)
    # make_vec_env handles Spine backend connection gracefully
    print(f"\nCreating {N_ENVS} training environment(s)...")
    train_env = make_vec_env(
        "Upkie-Spine-Servos",
        n_envs=N_ENVS,
        seed=SEED,
        env_kwargs=env_kwargs,
        wrapper_class=wrap_env,
    )
    
    print(f"\nAction space: {train_env.action_space}")
    print(f"Observation space: {train_env.observation_space}")
    
    # Setup checkpoint callback only (evaluation disabled - Spine backend timeouts)
    print("\nNote: Evaluation disabled due to Spine backend instability.")
    print("Models will be saved every 50k steps. Test manually after training.")
    
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,  # Save every 50k steps
        save_path="./models_servos/checkpoints/",
        name_prefix="ppo_upkie_servos",
        verbose=1,
    )
    
    # Create PPO model
    print("\nInitializing PPO model...")
    model = PPO(
        env=train_env,
        tensorboard_log="./logs_servos/tensorboard/",
        device="auto",
        **PPO_PARAMS
    )
    
    print(f"\nStarting training...")
    print(f"TensorBoard: tensorboard --logdir ./logs_servos/tensorboard/")
    print(f"Monitor progress in real-time!\n")
    
    # Train the model
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=[checkpoint_callback],
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
    
    # Save final model
    final_model_path = "./models_servos/ppo_upkie_servos_final"
    model.save(final_model_path)
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"Final model saved to: {final_model_path}.zip")
    print(f"Checkpoints saved in: ./models_servos/checkpoints/")
    print(f"\nTo test your trained models:")
    print(f"  # Test final model:")
    print(f"  python rollout_policy_servos.py --model {final_model_path}.zip --episodes 5")
    print(f"\n  # Test specific checkpoints:")
    print(f"  python rollout_policy_servos.py --model ./models_servos/checkpoints/ppo_upkie_servos_500000_steps.zip --episodes 5")
    print(f"\n  # Compare early vs late:")
    print(f"  python rollout_policy_servos.py --model ./models_servos/checkpoints/ppo_upkie_servos_200000_steps.zip --episodes 5")
    print("=" * 70)
    
    # Cleanup
    train_env.close()

if __name__ == "__main__":
    main()

