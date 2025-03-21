import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from huggingface_sb3 import package_to_hub
from custom_policy import Policy
import demo

env = gym.make("LunarLander-v2", render_mode="rgb_array")
observation, info = env.reset()

# Load Trained Model and apply extension:
model = PPO.load("base_policy")
model.policy = Policy(model.policy)

demo.run_demo(model)

package_to_hub(model,
               model_architecture='todo',
               env_id='todo',
               eval_env=DummyVecEnv([lambda:gym.make(env_id,render_mode='rgb_array')]),
               repo_id='todo',
               commit_message='test0')
