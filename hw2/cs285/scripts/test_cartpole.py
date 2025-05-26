#!/usr/bin/env python3

import gym
import numpy as np
import time
from gym import spaces

class CustomCartPole(gym.Env):
    def __init__(self):
        self.env = gym.make('CartPole-v0', render_mode='human')  # Set render_mode to 'human'
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.max_steps = 1000  # Increased max steps
        
    def reset(self):
        obs, _ = self.env.reset()
        self.steps = 0
        return obs, {}
        
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.steps += 1
        
        # Get pole angle from observation (index 2 is the pole angle)
        pole_angle = abs(obs[2])
        
        # Only terminate if pole angle exceeds 90 degrees (π/2 radians)
        if pole_angle > np.pi/2:
            terminated = True
        else:
            terminated = False
            
        # Truncate if max steps reached
        if self.steps >= self.max_steps:
            truncated = True
            
        return obs, reward, terminated, truncated, info
        
    def render(self):
        return self.env.render()
        
    def close(self):
        self.env.close()

class RightActionPolicy:
    def __init__(self, env):
        self.env = env
        
    def get_action(self, obs):
        # Always return action 1 (right) for CartPole
        return 1

def main():
    # Create custom environment
    env = CustomCartPole()
    policy = RightActionPolicy(env)
    
    # Run a few episodes
    for episode in range(5):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = policy.get_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            time.sleep(0.01)  # Slow down visualization
            
        print(f"Episode {episode + 1} finished with reward: {episode_reward}")
    
    env.close()

if __name__ == "__main__":
    main() 