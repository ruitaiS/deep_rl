import pygame
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor


def run_demo(model):

    env = gym.make("LunarLander-v2", render_mode="rgb_array")
    observation, info = env.reset()
    env = Monitor(gym.make("LunarLander-v2", render_mode='rgb_array'))

    # Demo The Trained Model:
    pygame.init()
    screen = pygame.display.set_mode((600, 400))
    observation, info = env.reset()
    episode_reward = 0
    episode_steps = 0
    for _ in range(10000):
        action, _ = model.predict(observation)

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
            print(f"Terminated: {terminated}, Truncated: {truncated}\n")
            #print("resetting environment")
            episode_reward = 0
            episode_steps = 0
            observation, info = env.reset()

    env.close()
    pygame.quit()

#run_demo(model=None, model_filename='base_policy')
#run_demo(model=None, model_filename='extended_policy')
