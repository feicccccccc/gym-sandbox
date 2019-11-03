# Reference: http://karpathy.github.io/2016/05/31/rl/

import gym
import numpy as np

env = gym.make('Pong-v0')
# print(env.action_space) # action space
# print(env.observation_space) # observation space

env.reset()
for episode in range(20):
    observation = env.reset()
    for frame in range(100):
        env.render()
        action = env.action_space.sample()

        # Action meaning
        # env.unwrapped.get_action_meanings()
        # ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']
        # Right = Up
        # Left = Down

        observation ,reward , done , info = env.step(action) # take a random action

        # observation : Pixel array of Pong, shape: (210,160,3)
        # reward : Reward of Pong, -1 indicate loss at the current frame, +1 indicate winning at current frame
        # done : If the current episode is finished
        # info : diagnostic information

        # env.step() <- Action
        # observation[0] <- Observation
        # observation[1] <- Reward

        print(frame
              ,observation.shape
              ,reward
              ,done
              ,info)
        if done:
            print("Episode finished after {} frame".format(frame + 1))
            break
env.close()