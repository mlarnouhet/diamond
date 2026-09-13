import gymnasium as gym
import ale_py
import cv2
import numpy as np

gym.register_envs(ale_py)
env = gym.make("ALE/Pong-v5", render_mode="human")
observation, info = env.reset(seed=42)

while True:
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
env.close()

print("Done !")





  




