import pygame
import gymnasium as gym


from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.monitor import Monitor

import demo


env = gym.make("LunarLander-v2", render_mode="rgb_array")
observation, info = env.reset()

"""
Observation Space Shape (8,):
- Horizontal pad coordinate (x)
- Vertical pad coordinate (y)
- Horizontal speed (x)
- Vertical speed (y)
- Angle
- Angular speed
- If the left leg contact point has touched the land (boolean)
- If the right leg contact point has touched the land (boolean)

Action Space Shape 4:
- Action 0: Do nothing,
- Action 1: Fire left orientation engine,
- Action 2: Fire the main engine,
- Action 3: Fire right orientation engine.
"""
print("_____OBSERVATION SPACE_____ \n")
print("Observation Space Shape", env.observation_space.shape)
print("\n _____ACTION SPACE_____ \n")
print("Action Space Shape", env.action_space.n)

'''
# Train and Save:
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
model.save("testbase")

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

# Load Trained Model
model = PPO.load("testbase")
demo.run_demo(model=model)

# Apply Extension and Re-demo
model.policy = Policy(model.policy)
demo.run_demo(model=model)
