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

import demo

env = gym.make("LunarLander-v2", render_mode="rgb_array")
observation, info = env.reset()

# Build on base policy:
class Policy(ActorCriticPolicy):
	def __init__(self, base_policy):
		super().__init__(base_policy.observation_space, base_policy.action_space, lr_schedule = lambda _ : 0.0003)
		self.load_state_dict(base_policy.state_dict())

	def forward(self, obs):
		return super().forward(obs)

	def forward_actor(self, obs):
		return super().forward_actor(obs)

	def forward_critic(self, obs):
		return super().forward_critic(obs)

	def predict(self, obs, state, episode_start, deterministic):
		action, _ = super().predict(obs, state, episode_start, deterministic)

		# Cut Engines on Touchdown
		if obs[6] or obs[7]:
			return (0, None)
		else:
			return action, None

# Load Trained Model and apply extension:
model = PPO.load("ppo-LunarLander-v2")
model.policy = Policy(model.policy)
model.save("extended_policy")

demo.run_demo(model=model)
