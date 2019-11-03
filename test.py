import gym
env = gym.make('Pong-v0')
env.reset()
for i in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
    print(i)
env.close()