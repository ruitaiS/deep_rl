import sys
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from extended_lander import ExtendedLunarLander


filename = sys.argv[1] if len(sys.argv) > 1 else '__v0.1'
timesteps = int(sys.argv[2]) if len(sys.argv) > 2 else 500000

#env = gym.make("LunarLander-v2", render_mode="rgb_array")
#observation, info = env.reset()

# Train and Save:
'''
rwd_func = {
        'x_pos_penalty': 0.0,         # Penalty for horizontal distance from center
        'x_vel_penalty': 0.0,         # Penalty for horizontal velocity
        'y_vel_penalty': 0.0,         # Penalty for vertical velocity
        'angle_penalty': 0.0,         # Penalty for tilt
        'angular_vel_penalty': 0.0,   # Penalty for spinning
        'main_engine_penalty': 0.0,   # Penalty when main engine is used
        'side_engine_penalty': 0.0,   # Penalty when side engines are used
        'idle_bonus': 0.0             # Bonus for doing nothing (action == 0)
        }
'''

env = make_vec_env(lambda: ExtendedLunarLander(rwd_func=None))
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

print(f"Training for {timesteps} timesteps && saving to '{filename}'")
model.learn(total_timesteps = timesteps)
model.save(filename)

# Evaluate Mean Reward of Current Model
eval_env = Monitor(gym.make("LunarLander-v2", render_mode='rgb_array'))
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
print(f"mean_reward: {mean_reward:.2f}, sd: {std_reward}")
