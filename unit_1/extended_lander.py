import gymnasium as gym
from gymnasium.envs.box2d.lunar_lander import LunarLander
import numpy as np

class ExtendedLunarLander(LunarLander):
    def __init__(self, rwd_func=None, **kwargs):
        super().__init__(**kwargs)

        # Default reward shaping values
        default_func = {
            'x_pos_pen': lambda x, y: -1*abs(x)/(y+1),         # Penalty for horizontal distance from center
            'x_pos_rew': lambda x, y: 1 / ((abs(x) + 1) * (y + 1)),         # Reward for horizontal closeness to center;
            'x_vel_penalty': 0.0,         # Penalty for horizontal velocity
            'y_vel_penalty': 0.0,         # Penalty for vertical velocity
            'angle_penalty': 0.0,         # Penalty for tilt
            'angular_vel_penalty': 0.0,   # Penalty for spinning
            #'main_engine_pen': lambda l_touch, r_touch, y, vy: -200*int(l_touch or r_touch),   # Penalty when main engine is used unnecessarily
            'engine_off_rew': lambda l_touch, r_touch, action, x: (1*int(l_touch) + 1*int(r_touch))*int(action == 0)*(1/(abs(x)+1)),   # Reward to cut engines when landing
            'side_engine_penalty': 0.0,   # Penalty when side engines are used
            'idle_bonus': 0.0             # Bonus for doing nothing (action == 0)
        }

        self.config = default_func if rwd_func is None else {**default_func, **rwd_func}
        self.prev_action = None

    def step(self, action):
        obs, base_reward, terminated, truncated, info = super().step(action)

        # Extract physics state
        x = self.lander.position.x
        y = self.lander.position.y
        vx = self.lander.linearVelocity.x
        vy = self.lander.linearVelocity.y
        theta = self.lander.angle
        vtheta = self.lander.angularVelocity
        l_touch = self.legs[0].ground_contact
        r_touch = self.legs[1].ground_contact

        #print(f"Y_pos: {y}")
        #print(f"x_pos: {x} | y_pos: {y} | v_x: {vx} | v_y: {vy}")

        # Reward shaping
        shaped_reward = 0.0

        shaped_reward += self.config['x_pos_pen'](x, y)
        shaped_reward += self.config['x_pos_rew'](x, y)

        #shaped_reward += self.config['main_engine_pen'] (l_touch, r_touch, y, vy)
        #shaped_reward += self.config['engine_off_rew'] (l_touch, r_touch, action, x)
        
        #shaped_reward -= self.config['x_vel_penalty'] * abs(vx)
        #shaped_reward -= self.config['y_vel_penalty'] * abs(vy)
        #shaped_reward -= self.config['angle_penalty'] * abs(theta)
        #shaped_reward -= self.config['angular_vel_penalty'] * abs(vtheta)

        #print(f"Shaped Reward: {shaped_reward}")

        '''
        # Engine usage penalties
        if action == 1 or action == 3:  # side engines
            shaped_reward -= self.config['side_engine_penalty']
        elif action == 2:  # main engine
            shaped_reward -= self.config['main_engine_penalty']

        # Idle bonus
        if action == 0:
            shaped_reward += self.config['idle_bonus']
        '''

        # Combine total reward
        total_reward = base_reward + shaped_reward

        
        info.update({
            'shaped_acc': info.get('shaped_acc', 0) + shaped_reward
        })

        return obs, total_reward, terminated, truncated, info
