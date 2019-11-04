# Reference: http://karpathy.github.io/2016/05/31/rl/

import gym
import numpy as np
import pickle
import misc_func as func

# Hyperparameter
H = 200 # number of hidden layer neurons
batch_size = 10 # every how many episodes to do a param update?
learning_rate = 1e-4
resume = True # resume from previous checkpoint?
render = True # Render to show information or not?
gamma = 0.99 # discount factor
decay_rate = 0.99 # for RMSprop

# Model init
D = 80 * 80 # input dimensionality: 80x80 grid

# gradient calculation
xs,hs,dlogps,drs = [],[],[],[]
running_reward = None

# reward
reward_sum = 0

# create a dict to store the parameter
if resume:
  model = pickle.load(open('save.p', 'rb'))
else:
  model = {}
  model['W1'] = np.random.randn(H,D) / np.sqrt(D) # "Xavier" initialization
  model['W2'] = np.random.randn(H) / np.sqrt(H)

grad_buffer = { k : np.zeros_like(v) for k,v in model.items() } # update buffers that add up gradients over a batch
rmsprop_cache = { k : np.zeros_like(v) for k,v in model.items() } # rmsprop memory

# Model:
# Input Layer: (6400,200)
# hidden Layer: (200,1)
# ouput : Probability of choosing up / down

env = gym.make('Pong-v0')
observation = env.reset()

# func.plot_cur_image(observation)
# preprocess: crop the gaming area for 60 * 60 image
# show what happen during the process

prev_x = None # used in computing the difference frame

episode_number = 0

while True:

    if render: env.render()

    # preprocess the observation, set input to network to be difference image
    cur_x = func.prepro(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros(D)
    prev_x = cur_x

    #action = env.action_space.sample()
    # Action meaning
    # env.unwrapped.get_action_meanings()
    # ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']
    # Right = Up
    # Left = Down

    # ------
    # NN forward Prop
    # ------

    # forward the policy network and sample an action from the returned probability
    aprob, h = func.policy_forward(x,model)
    action = 2 if np.random.uniform() < aprob else 3  # roll the dice!

    # ------
    # NN Back Prop
    # ------

    # record various intermediates (needed later for backprop)
    xs.append(x)  # observation
    hs.append(h)  # hidden state
    y = 1 if action == 2 else 0  # intermediate label, 1 for up and 0 for down. Binary classification problem
    dlogps.append(y - aprob) # cross-entropy loss gradient

    observation ,reward , done , info = env.step(action) # take a random action

    # observation : Pixel array of Pong, shape: (210,160,3)
    # reward : Reward of Pong, -1 indicate loss at the current frame, +1 indicate winning at current frame
    # done : If the current episode is finished
    # info : diagnostic information

    # env.step() <- Action
    # observation[0] <- Observation
    # observation[1] <- Reward

    reward_sum += reward

    drs.append(reward)  # record reward (has to be done after we call step() to get reward for previous action)


    if done:  # an episode finished
        # Finish 1 game: 1 side get 21 points
        # maybe it win
        # or maybe it loss
        # for some action

        episode_number += 1

        # stack together all inputs, hidden states, action gradients, and rewards for this episode
        # ------
        # NN Back Prop
        # ------
        epx = np.vstack(xs) # activation of layer 1
        eph = np.vstack(hs) # activation of layer 2 (hidden layer)
        epdlogp = np.vstack(dlogps) # activation of output layer
        epr = np.vstack(drs) # reward
        xs, hs, dlogps, drs = [], [], [], []  # reset array memory

        # compute the discounted reward backwards through time
        discounted_epr = func.discount_rewards(epr,gamma)
        # standardize the rewards to be unit normal (helps control the gradient estimator variance)
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)

        epdlogp *= discounted_epr  # modulate the gradient with advantage (PG magic happens right here.)
        grad = func.policy_backward(eph, epx, epdlogp, model)

        for k in model: grad_buffer[k] += grad[k]  # accumulate grad over batch

        # perform rmsprop parameter update every batch_size episodes
        if episode_number % batch_size == 0:
            for k, v in model.items():
                g = grad_buffer[k]  # gradient

                # Update parameters
                rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g ** 2
                model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)


                grad_buffer[k] = np.zeros_like(v)  # reset batch gradient buffer

        # book-keeping
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01

        print('resetting env. episode reward total was {}. running mean: {}'.format(reward_sum, running_reward))

        if episode_number % 100 == 0:
            pickle.dump(model, open('save.p', 'wb'))
        reward_sum = 0
        observation = env.reset()  # reset env
        prev_x = None

    if reward != 0:  # Pong has either +1 or -1 reward exactly when game ends.
        print('ep {}: game finished, reward: {}'.format(episode_number, reward) + ('' if reward == -1 else ' !!!!!!!!'))

    # print(episode_number
    #       ,observation.shape
    #       ,reward
    #       ,done
    #       ,info)



env.close()