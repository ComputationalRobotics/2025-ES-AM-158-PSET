#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gymnasium as gym
import numpy as np
from upkie.logging import logger


class ServosRewardWrapper(gym.Wrapper):
    """
    Reward wrapper for Upkie-Servos environment.
    
    Provides reward shaping to encourage the robot to:
    - Stay upright (minimize pitch angle)
    - Minimize angular velocity (stable motion)
    - Stay near starting position (avoid excessive drift)
    - Penalize falls and excessive drift
    """
    
    def __init__(self, env, fall_pitch: float = 1.0, max_ground_position: float = 5.0):
        """
        Initialize reward wrapper.
        
        Args:
            env: The base Upkie-Servos environment
            fall_pitch: Maximum pitch angle before fall detection (radians)
            max_ground_position: Maximum ground position before termination (meters)
        """
        super().__init__(env)
        self.fall_pitch = fall_pitch
        self.max_ground_position = max_ground_position
        
    def reset(self, **kwargs):
        """Reset the environment."""
        return self.env.reset(**kwargs)
    
    def step(self, action):
        """
        Execute action and compute shaped reward.
        
        Args:
            action: Action from the policy
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Extract spine observation for reward calculation
        spine_obs = info.get("spine_observation", {})
        
        # Get base orientation
        base_ori = spine_obs.get("base_orientation", {})
        pitch = base_ori.get("pitch", 0.0)
        angular_velocity = base_ori.get("angular_velocity", [0, 0, 0])
        pitch_rate = angular_velocity[1]  # Angular velocity around Y axis (pitch)
        
        # Get wheel odometry
        wheel_odom = spine_obs.get("wheel_odometry", {})
        ground_pos = wheel_odom.get("position", 0.0)
        ground_vel = wheel_odom.get("velocity", 0.0)
        
        # ===== Reward Shaping =====
        
        # 1. Base reward: survival bonus
        reward = 1.0
        
        # 2. Penalize deviation from upright position (most important!)
        reward -= 0.5 * pitch**2
        
        # 3. Penalize high angular velocity (encourages smooth motion)
        reward -= 0.1 * pitch_rate**2
        
        # 4. Penalize excessive ground velocity (avoid racing away)
        reward -= 0.01 * ground_vel**2
        
        # 5. Penalize drift from origin (stay near starting position)
        reward -= 0.005 * ground_pos**2
        
        # ===== Termination Conditions =====
        
        # Detect fall based on pitch angle
        if abs(pitch) > self.fall_pitch:
            terminated = True
            reward -= 10.0  # Large penalty for falling
            logger.warning(
                f"Fall detected (pitch={abs(pitch):.2f} rad, fall_pitch={self.fall_pitch:.2f} rad)"
            )
        
        # Detect excessive drift
        if abs(ground_pos) > self.max_ground_position:
            terminated = True
            reward -= 5.0  # Penalty for drifting too far
            logger.warning(
                f"Excessive drift detected (ground_pos={abs(ground_pos):.2f} m, max={self.max_ground_position:.2f} m)"
            )
        
        return obs, reward, terminated, truncated, info

