import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from huggingface_sb3 import package_to_hub
from custom_policy import Policy
import demo

env = gym.make("LunarLander-v2", render_mode="rgb_array")
observation, info = env.reset()

# Load Trained Model and apply extension:
model = PPO.load("__base_policy")
model.policy = Policy(model.policy)
show_demo = True
upload = False

if show_demo:
    demo.run_demo(model)

if upload:
    package_to_hub(model_name='lander_model',
                   model=model,
                   model_architecture='PPO',
                   env_id='LunarLander-v2',
                   eval_env=DummyVecEnv([lambda:gym.make('LunarLander-v2',render_mode='rgb_array')]),
                   repo_id='shaoruitai/lander_model',
                   commit_message='v0.1')
