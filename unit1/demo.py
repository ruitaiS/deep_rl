import pygame
import gymnasium as gym

from huggingface_sb3 import load_from_hub, package_to_hub
from huggingface_hub import notebook_login

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.monitor import Monitor

import torch
import torch.nn as nn
import numpy as np

def run_demo(model, model_filename=None):

    env = gym.make("LunarLander-v2", render_mode="rgb_array")
    observation, info = env.reset()
    if not model:
        model = PPO.load(model_filename)
    env = Monitor(gym.make("LunarLander-v2", render_mode='rgb_array'))

    # Demo The Trained Model:
    pygame.init()
    screen = pygame.display.set_mode((600, 400))
    observation, info = env.reset()
    episode_reward = 0
    episode_steps = 0
    for _ in range(10000):
        action, _ = model.predict(observation)

        '''
        - Horizontal pad coordinate (x)
        - Vertical pad coordinate (y)
        - Horizontal speed (x) 2
        - Vertical speed (y) 3
        - Angle 4
        - Angular speed 5
        - If the left leg contact point has touched the land (boolean) 6
        - If the right leg contact point has touched the land (boolean) 7
        '''


        #action = env.action_space.sample()
        #print(f"Action taken:{action}")

        # Next step as a result of sampled action
        observation, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        episode_steps += 1
        #print(f"Observation: {observation}, Reward: {reward}, Info: {info}")

        frame = env.render()
        if frame is not None:
            # Convert Gym frame to Pygame format
            surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
            screen.blit(surface, (0, 0))
            pygame.display.flip()
        else:
            print("⚠️ Warning: Frame is None or not an array!")

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                break

        if terminated or truncated:
            print(f"Total Reward: {episode_reward}, Total Steps: {episode_steps}")
            print(f"Terminated: {terminated}, Truncated: {truncated}")
            print("resetting environment")
            episode_reward = 0
            episode_steps = 0
            observation, info = env.reset()

    env.close()
    pygame.quit()
