import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from extended_lander import ExtendedLunarLander

#env = gym.make("LunarLander-v2", render_mode="rgb_array")
#observation, info = env.reset()

# Train and Save:
#env = make_vec_env('LunarLander-v2', n_envs=16)
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

env = make_vec_env(lambda: ExtendedLunarLander(rwd_func=rwd_func))
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

model.learn(total_timesteps = 1500000)
model.save("__v0.1b")

# Evaluate Mean Reward of Current Model
eval_env = Monitor(gym.make("LunarLander-v2", render_mode='rgb_array'))
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, deterministic=True)
print(f"mean_reward: {mean_reward:.2f}, sd: {std_reward}")
