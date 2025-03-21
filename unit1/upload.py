import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv

from huggingface_sb3 import package_to_hub

repo_id = 'todo'
env_id = 'todo'
eval_env = DummyVecEnv([lambda: gym.make(env_id, render_mode="rgb_array")])
model_architecture = 'todo'
commit_message = 'test0'

package_to_hub(model = model,
               model_name = model_name,
               model_architecture = model_architecture,
               env_id = env_id,
               eval_env = eval_env,
               repo_id = repo_id,
               commit_message = commit_message)
