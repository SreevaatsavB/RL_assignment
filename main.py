import numpy as np
import matplotlib.pyplot as plt
import gym
from mmWave_bandits import mmWaveEnv
from matplotlib import pyplot as plt
import seaborn as sns

# class EpsGreedyQPolicy():
#     """Implement the epsilon greedy policy

#     Eps Greedy policy either:

#     - takes a random action with probability epsilon
#     - takes current best action with prob (1 - epsilon)
#     """
#     def __init__(self, eps=.1):
#         self.eps = eps

#     def select_action(self, q_values):
#         """Return the selected action

#         # Arguments
#             q_values (np.ndarray): List of the estimations of Q for each action

#         # Returns
#             Selection action
#         """
#         assert q_values.ndim == 1
#         nb_actions = q_values.shape[0]

#         if np.random.uniform() < self.eps:
#             action = np.random.randint(0, nb_actions)
#         else:
#             action = np.argmax(q_values)
#         return action

# def lin_epsilon(epsilon, env):
#     x = np.random.uniform()
#     # if x <= epsilon:
#     print(env.action_space)


# env = mmWaveEnv()
# lin_epsilon(0.5, env)

# Define your custom mmWave environment
env = mmWaveEnv()

# Number of actions (combination of beam type and beam number)
num_actions = (2, env.Nbeams)  # A tuple of two values: (beam type, beam number)

# Initialize Q-values for each action
# Q = np.zeros(num_actions)
Q = {}

# Linear schedule for epsilon
initial_epsilon = 0.99
final_epsilon = 0.05
# epsilon_decay_steps = 1000  # Adjust this based on your training schedule
epsilon = initial_epsilon

# Training parameters
num_episodes = 1000  # Adjust as needed

epsilon_decay_steps = num_episodes
max_time_steps = env.Horizon


# max_action = 0
# max_action_rew = 0

rew_arr = [0]

for episode in range(num_episodes):
    max_action = (0,0)
    max_action_rew = 0


    obs = env.reset()
    total_reward = 0

    for t in range(max_time_steps):
        # Epsilon-greedy action selection
        if np.random.rand() < epsilon:
            # EXPLORE
            # Randomly choose action type and beam number
            action = (np.random.randint(2), np.random.randint(env.Nbeams))
        else:
            # EXPLOIT
            # Choose the action with the highest Q-value
            action = max_action
            # print("MAX ACTION = ", max_action)

            # print("Action = ", action)
            # print()
            # print("Q = ", Q)
            # print()
            # break
            # action = np.unravel_index(np.argmax(Q), (2, 10))
            # print("Q argmax: ", np.argmax(Q))
            
        new_obs, reward, correct, terminated, truncated, _ = env.step(action)

        if action not in Q:
            Q[action] = [1, reward]

            if max_action_rew <= Q[action][1]:
                max_action_rew = Q[action][1]
                max_action = action
        else:
            Q[action][0]+=1
            if correct==1:
                Q[action][1]+=(1/(Q[action][0]))*(reward-Q[action][1])

            if max_action_rew <= Q[action][1]:
                max_action_rew = Q[action][1]
                max_action = action
        
        # print(Q)
        # break

        # Update Q-value for the chosen action
        
        # Q[action] = Q[action] + 1/(action_dic[action]) * (reward - Q[action])

        total_reward += reward

        obs = new_obs

        if terminated:
            break
        
   
    print(Q)
    # rew_arr.append(total_reward + rew_arr[-1])
    rew_arr.append(total_reward)

    print()
    # Decay epsilon over time
    # epsilon = np.random.uniform(0,1,1)[0]
    epsilon = max(final_epsilon, initial_epsilon - episode / epsilon_decay_steps)
    print(epsilon)

    # Print the total reward for this episode
    print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")



plt.plot(range(1,len(rew_arr)+1),rew_arr)
plt.show()
