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

env = gym.make("LunarLander-v2", render_mode="rgb_array")
observation, info = env.reset()

print("_____OBSERVATION SPACE_____ \n")
print("Observation Space Shape", env.observation_space.shape)
print("Sample observation", env.observation_space.sample()) # Get a random observation

"""Observation Space Shape (8,):env = Monitor(gym.make("LunarLander-v2", render_mode='rgb_array'))
- Horizontal pad coordinate (x)
- Vertical pad coordinate (y)
- Horizontal speed (x)
- Vertical speed (y)
- Angle
- Angular speed
- If the left leg contact point has touched the land (boolean)
- If the right leg contact point has touched the land (boolean)
"""

print("\n _____ACTION SPACE_____ \n")
print("Action Space Shape", env.action_space.n)
print("Action Space Sample", env.action_space.sample()) # Take a random action

"""Action Space Shape 4:
- Action 0: Do nothing,
- Action 1: Fire left orientation engine,
- Action 2: Fire the main engine,
- Action 3: Fire right orientation engine.
"""

# Train and Save:
'''
env = make_vec_env('LunarLander-v2', n_envs=16)
model = PPO(
    policy = 'MlpPolicy',
    env = env,
    n_steps = 1024,
    batch_size = 64,
    n_epochs = 4,
    gamma = 0.999,
    gae_lambda = 0.98,
    ent_coef = 0.01,
    verbose=1)

model.learn(total_timesteps = 500000)
model.save("ppo-LunarLander-v2")

eval_env = Monitor(gym.make("LunarLander-v2", render_mode='rgb_array'))
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
print(f"mean_reward: {mean_reward:.2f}, sd: {std_reward}")
'''

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
model.save("ppo-LunarLander-v2-modified")

# Load Final Model:
#model = PPO.load("ppo-LunarLander-v2-modified")
env = Monitor(gym.make("LunarLander-v2", render_mode='rgb_array'))

# Demo The Trained Model:
pygame.init()
screen = pygame.display.set_mode((600, 400))
observation, info = env.reset()
episode_reward = 0
episode_steps = 0
for _ in range(10000):
	action, _ = model.predict(observation)

	#if observation[6] and observation[7]:
	#	action = 0



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

'''
env = gym.make("LunarLander-v2")
env.reset()


#### Vectorized Environment
# Create a vectorized environment
# (stacking 16 independent environments into a single environment)
env = make_vec_env('LunarLander-v2', n_envs=16)

model = PPO(
    policy = 'MlpPolicy',
    env = env,
    n_steps = 1024,
    batch_size = 64,
    n_epochs = 4,
    gamma = 0.999,
    gae_lambda = 0.98,
    ent_coef = 0.01,
    verbose=1)

model.learn(total_timesteps = 500000)
model.save("ppo-LunarLander-v2")

eval_env = Monitor(gym.make("LunarLander-v2", render_mode='rgb_array'))
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
print(f"mean_reward: {mean_reward:.2f}, sd: {std_reward}")



"""Reward function (the function that will give a reward at each timestep) 💰:

After every step a reward is granted. The total reward of an episode is the **sum of the rewards for all the steps within that episode**.

For each step, the reward:

- Is increased/decreased the closer/further the lander is to the landing pad.
-  Is increased/decreased the slower/faster the lander is moving.
- Is decreased the more the lander is tilted (angle not horizontal).
- Is increased by 10 points for each leg that is in contact with the ground.
- Is decreased by 0.03 points each frame a side engine is firing.
- Is decreased by 0.3 points each frame the main engine is firing.

The episode receive an **additional reward of -100 or +100 points for crashing or landing safely respectively.**

An episode is **considered a solution if it scores at least 200 points.**
"""

'''