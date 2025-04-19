import gymnasium as gym
from gymnasium.envs.box2d.lunar_lander import LunarLander
import numpy as np

class ExtendedLunarLander(LunarLander):
    def __init__(self, rwd_func=None, **kwargs):
        super().__init__(**kwargs)

        # Default reward shaping values
        default_func = {
            'x_pos_penalty': 0.0,         # Penalty for horizontal distance from center
            'x_vel_penalty': 0.0,         # Penalty for horizontal velocity
            'y_vel_penalty': 0.0,         # Penalty for vertical velocity
            'angle_penalty': 0.0,         # Penalty for tilt
            'angular_vel_penalty': 0.0,   # Penalty for spinning
            'main_engine_penalty': 0.0,   # Penalty when main engine is used
            'side_engine_penalty': 0.0,   # Penalty when side engines are used
            'idle_bonus': 0.0             # Bonus for doing nothing (action == 0)
        }

        self.config = default_func if rwd_func is None else {**default_func, **rwd_func}
        self.prev_action = None

    def step(self, action):
        obs, base_reward, terminated, truncated, info = super().step(action)
        done = terminated or truncated  # Optional if needed for logic

        # Extract physics state
        x = self.lander.position.x
        y = self.lander.position.y
        vx = self.lander.linearVelocity.x
        vy = self.lander.linearVelocity.y
        theta = self.lander.angle
        theta_dot = self.lander.angularVelocity
        left_contact = self.legs[0].ground_contact
        right_contact = self.legs[1].ground_contact

        # Reward shaping
        shaped_reward = 0.0

        shaped_reward -= self.config['x_pos_penalty'] * abs(x)
        shaped_reward -= self.config['x_vel_penalty'] * abs(vx)
        shaped_reward -= self.config['y_vel_penalty'] * abs(vy)
        shaped_reward -= self.config['angle_penalty'] * abs(theta)
        shaped_reward -= self.config['angular_vel_penalty'] * abs(theta_dot)

        # Engine usage penalties
        if action == 1 or action == 3:  # side engines
            shaped_reward -= self.config['side_engine_penalty']
        elif action == 2:  # main engine
            shaped_reward -= self.config['main_engine_penalty']

        # Idle bonus
        if action == 0:
            shaped_reward += self.config['idle_bonus']

        # Combine total reward
        total_reward = base_reward + shaped_reward

        # Optional logging in info dict
        info.update({
            'base_reward': base_reward,
            'shaped_reward': shaped_reward,
            'total_reward': total_reward
        })

        return obs, base_reward, terminated, truncated, info
