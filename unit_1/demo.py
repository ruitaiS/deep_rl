import sys
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

        x = observation[0]
        y = observation[1]
        vx = observation[2]
        vy = observation[3]
        theta = observation[4]
        vtheta = observation[5]
        l_touch = observation[6]
        r_touch = observation[7]


        #print(f"x: {x:.2f}, y: {y:.2f}, vx: {vx:.2f}, vy: {vy:.2f}, theta: {theta:.2f}, vtheta: {vtheta:.2f}\
        #      \naction: {action}, l_touch: {l_touch:.2f}, r_touch: {r_touch:.2f} \nReward: {reward:.2f}")

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
            print(f"x: {x:.2f}, y: {y:.2f}")
            print(f"Total Reward: {episode_reward}, Total Steps: {episode_steps}")
            print(f"Terminated: {terminated}, Truncated: {truncated}\n")
            #print("resetting environment")
            episode_reward = 0
            episode_steps = 0
            observation, info = env.reset()

    env.close()
    pygame.quit()

filename = sys.argv[1] if len(sys.argv) > 1 else '__v0.1'
model = PPO.load(filename)
run_demo(model)