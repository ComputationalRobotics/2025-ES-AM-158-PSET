#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import sys
from typing import Dict, Optional

import numpy as np
import gymnasium as gym
import upkie.envs
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO

# ---------------------------------------------------------------------
# Upkie registration
# ---------------------------------------------------------------------
upkie.envs.register()

EPS = 1e-6  # avoid torque clamp spam

# ---------------------------------------------------------------------
# Wrappers (same semantics as your training snippet)
#   - Wheels: velocity command in [-1,1] -> scaled to vel limits
#   - Hips/Knees: position command in [-1,1] -> mapped to pos limits
#   Output per-joint dict matches UpkieServos API (no top-level "servo" key)
# ---------------------------------------------------------------------
class ServoVelActionWrapper(gym.ActionWrapper):
    def __init__(
        self,
        env: gym.Env,
        fixed_order: Optional[Dict[str, list]] = None,
        gains: Optional[Dict[str, float]] = None,
    ):
        super().__init__(env)
        if not isinstance(self.env.action_space, gym.spaces.Dict):
            raise TypeError("UpkieServos expected Dict action_space")

        # Discover joints
        all_names = list(self.env.action_space.spaces.keys())
        wheels = [n for n in all_names if "wheel" in n]
        legs   = [n for n in all_names if ("hip" in n) or ("knee" in n)]

        # Freeze order if provided
        if fixed_order is not None:
            self.wheel_names = list(fixed_order["wheel_names"])
            self.leg_names   = list(fixed_order["leg_names"])
        else:
            self.wheel_names = wheels
            self.leg_names   = legs

        # Wheels limits (velocity, torque)
        self.wheel_vel_lim = np.array(
            [float(self.env.action_space[j]["velocity"].high[0]) for j in self.wheel_names],
            dtype=np.float32,
        )
        self.wheel_tau_lim = np.array(
            [float(self.env.action_space[j]["maximum_torque"].high[0]) for j in self.wheel_names],
            dtype=np.float32,
        )

        # Legs limits (position, torque)
        self.leg_pos_low  = np.array(
            [float(self.env.action_space[j]["position"].low[0]) for j in self.leg_names],
            dtype=np.float32,
        )
        self.leg_pos_high = np.array(
            [float(self.env.action_space[j]["position"].high[0]) for j in self.leg_names],
            dtype=np.float32,
        )
        self.leg_tau_lim = np.array(
            [float(self.env.action_space[j]["maximum_torque"].high[0]) for j in self.leg_names],
            dtype=np.float32,
        )

        # Controller gains (keep identical to training)
        gains = gains or {}
        self.kp_wheel = float(gains.get("kp_wheel", 0.0))
        self.kd_wheel = float(gains.get("kd_wheel", 1.7))
        self.kp_leg   = float(gains.get("kp_leg",   2.0))
        self.kd_leg   = float(gains.get("kd_leg",   1.7))

        # Action vector = [ wheels_vel  |  legs_pos ]
        n = len(self.wheel_names) + len(self.leg_names)
        self._n_wheels = len(self.wheel_names)
        self._n_legs   = len(self.leg_names)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(n,), dtype=np.float32)

    def action(self, action):
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        a = np.clip(a, -1.0, 1.0)

        # Split
        a_wheel = a[: self._n_wheels] if self._n_wheels > 0 else np.zeros(0, np.float32)
        a_leg   = a[self._n_wheels :]  if self._n_legs   > 0 else np.zeros(0, np.float32)

        # Wheels -> velocity
        wheel_v_cmd = a_wheel * self.wheel_vel_lim

        # Legs -> position (map [-1,1] -> [low, high])
        leg_pos_cmd = (
            self.leg_pos_low + (a_leg + 1.0) * 0.5 * (self.leg_pos_high - self.leg_pos_low)
            if self._n_legs > 0 else np.zeros(0, np.float32)
        )

        env_action = {}

        # Wheels
        for i, name in enumerate(self.wheel_names):
            env_action[name] = dict(
                position=np.nan,
                velocity=float(wheel_v_cmd[i]),
                feedforward_torque=0.0,
                kp_scale=self.kp_wheel,
                kd_scale=self.kd_wheel,
                maximum_torque=float(self.wheel_tau_lim[i] - EPS),
            )

        # Legs
        for i, name in enumerate(self.leg_names):
            env_action[name] = dict(
                position=float(leg_pos_cmd[i]),
                velocity=0.0,
                feedforward_torque=0.0,
                kp_scale=self.kp_leg,
                kd_scale=self.kd_leg,
                maximum_torque=float(self.leg_tau_lim[i] - EPS),
            )

        return env_action


class ServoObsFlattenWrapper(gym.ObservationWrapper):
    """Dict obs -> flat Box: [pos_0, vel_0, pos_1, vel_1, ...] in fixed joint order."""
    def __init__(self, env: gym.Env):
        super().__init__(env)
        if not isinstance(self.env.observation_space, gym.spaces.Dict):
            raise TypeError("UpkieServos expected Dict observation_space")

        # Preserve order from the wrapped env's action wrapper
        self.joint_names = list(self.env.observation_space.spaces.keys())

        lows, highs = [], []
        for j in self.joint_names:
            pos_box = self.env.observation_space[j]["position"]
            vel_box = self.env.observation_space[j]["velocity"]
            lows  += [float(pos_box.low[0]),  float(vel_box.low[0])]
            highs += [float(pos_box.high[0]), float(vel_box.high[0])]
        self.observation_space = gym.spaces.Box(
            low=np.asarray(lows, dtype=np.float32),
            high=np.asarray(highs, dtype=np.float32),
            dtype=np.float32,
        )

    def observation(self, observation):
        vec = []
        for j in self.joint_names:
            vec += [float(observation[j]["position"][0]),
                    float(observation[j]["velocity"][0])]
        return np.asarray(vec, dtype=np.float32)

# ---------------------------------------------------------------------
# Env factory
# ---------------------------------------------------------------------
def make_wrapped_env(
    *,
    frequency_hz: float = 200.0,
    max_steps: int = 300,
    fixed_order: Optional[Dict[str, list]] = None,
    gains: Optional[Dict[str, float]] = None,
) -> gym.Env:
    env = gym.make("Upkie-Spine-Servos", frequency=frequency_hz)
    env = ServoVelActionWrapper(env, fixed_order=fixed_order, gains=gains)
    # IMPORTANT: flatten after action wrapper so the observation order matches the action order
    env = ServoObsFlattenWrapper(env)
    env = TimeLimit(env, max_episode_steps=max_steps)
    return env

# ---------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------
def rollout(
    model_path: str,
    episodes: int,
    deterministic: bool,
    seed: Optional[int],
    signature_path: Optional[str],
):

    # Defaults (match training)
    frequency_hz = 200.0
    max_steps = 300
    fixed_order = None
    gains = dict(kp_wheel=0.0, kd_wheel=1.7, kp_leg=2.0, kd_leg=1.7)

    env = make_wrapped_env(
        frequency_hz=frequency_hz,
        max_steps=max_steps,
        fixed_order=fixed_order,
        gains=gains,
    )

    # Load model and bind to this (non-Vec) env
    model = PPO.load(model_path, env=env, device="auto", print_system_info=False)
    print(f"[rollout] Loaded model from: {model_path}")

    # Seed for reproducibility
    if seed is not None:
        try:
            env.reset(seed=seed)
            np.random.seed(seed)
        except Exception:
            pass

    for ep in range(1, episodes + 1):
        obs, info = env.reset()
        done = False
        ep_ret, ep_len = 0.0, 0

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_ret += float(reward)
            ep_len += 1
            done = bool(terminated or truncated)

        print(f"[Episode {ep}] return = {ep_ret:.3f}   len = {ep_len} steps")

    env.close()

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Rollout SB3 PPO policy on Upkie-Spine-Servos.")
    p.add_argument("--model", type=str, required=True,
                   help="Path to .zip model file (e.g., ./models/ppo_upkie_servos_final.zip)")
    p.add_argument("--episodes", type=int, default=3, help="Number of evaluation episodes")
    p.add_argument("--deterministic", action="store_true", help="Use deterministic actions")
    p.add_argument("--seed", type=int, default=None, help="Seed for env reset")
    p.add_argument("--signature", type=str, default=None,
                   help="Optional env_signature.json to lock joint order/gains")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()

    if not os.path.isfile(args.model):
        print(f"Model not found: {args.model}", file=sys.stderr)
        sys.exit(1)

    rollout(
        model_path=args.model,
        episodes=args.episodes,
        deterministic=bool(args.deterministic),
        seed=args.seed,
        signature_path=args.signature,
    )
    
    
## a simple run is
## python rollout_policy_servos.py --model ./models/serves_best/best_model.zip --episodes 5
