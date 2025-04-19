import sys
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from huggingface_sb3 import package_to_hub

filename = sys.argv[1] if len(sys.argv) > 1 else '__v0.1'
model = PPO.load(filename)
package_to_hub(model_name='lander_model',
                model=model,
                model_architecture='PPO',
                env_id='LunarLander-v2',
                eval_env=DummyVecEnv([lambda:gym.make('LunarLander-v2',render_mode='rgb_array')]),
                repo_id='shaoruitai/lander_model',
                commit_message=filename[2:])
