# import numpy as np
# import matplotlib.pyplot as plt
# import gym
# from mmWave_bandits import mmWaveEnv
# from matplotlib import pyplot as plt
# import seaborn as sns


# # Define your custom mmWave environment
# env = mmWaveEnv()

# # Number of actions (combination of beam type and beam number)
# num_actions = (2, env.Nbeams)  # A tuple of two values: (beam type, beam number)

# # Initialize Q-values for each action
# # Q = np.zeros(num_actions)
# Q = {}

# # Linear schedule for epsilon
# initial_epsilon = 0.80
# final_epsilon = 0.1
# # epsilon_decay_steps = 1000  # Adjust this based on your training schedule
# epsilon = initial_epsilon

# # Training parameters
# num_epochs = 10  # Adjust as needed

# epsilon_decay_steps = num_epochs
# max_time_steps = env.Horizon


# # max_action = 0
# # max_action_rew = 0

# rew_arr = []
# correct_arr = []

# for epoch in range(num_epochs):
#     max_action = (0,0)
#     max_action_rew = 0

#     tot_corr = 0
#     obs = env.reset()
#     total_reward = 0

#     for t in range(max_time_steps):
#         # Epsilon-greedy action selection
#         if np.random.rand() < epsilon:
#             # EXPLORE
#             # Randomly choose action type and beam number
#             action = (np.random.randint(2), np.random.randint(env.Nbeams))
#         else:
#             # EXPLOIT
#             # Choose the action with the highest Q-value
#             action = max_action
#             # print("MAX ACTION = ", max_action)

#             # print("Action = ", action)
#             # print()
#             # print("Q = ", Q)
#             # print()
#             # break
#             # action = np.unravel_index(np.argmax(Q), (2, 10))
#             # print("Q argmax: ", np.argmax(Q))
            
#         new_obs, reward, correct, terminated, truncated, _ = env.step(action)

#         tot_corr += correct

#         if action not in Q:
#             Q[action] = [1, reward]

#             if max_action_rew <= Q[action][1]:
#                 max_action_rew = Q[action][1]
#                 max_action = action
#         else:
#             Q[action][0]+=1
#             if correct==1:
#                 Q[action][1]+=(1/(Q[action][0]))*(reward-Q[action][1])

#             if max_action_rew <= Q[action][1]:
#                 max_action_rew = Q[action][1]
#                 max_action = action
        
#         # print(Q)
#         # break

#         # Update Q-value for the chosen action
        
#         # Q[action] = Q[action] + 1/(action_dic[action]) * (reward - Q[action])

#         total_reward += reward

#         obs = new_obs

#         if terminated:
#             break
        
#     correct_arr.append(tot_corr)
#     # print(Q)
#     # rew_arr.append(total_reward + rew_arr[-1])
#     rew_arr.append(total_reward/max_time_steps)

#     print()
#     # Decay epsilon over time
#     # epsilon = np.random.uniform(0,1,1)[0]
#     epsilon = max(final_epsilon, initial_epsilon - epoch / epsilon_decay_steps)
#     print(epsilon)

#     # Print the total reward for this epoch
#     print(f"Epoch {epoch + 1}/{num_epochs}, Total Reward: {total_reward}")


# plt.figure(figsize=(20,10))
# plt.subplot(1,2,1)
# plt.title("Avg Reward")
# plt.plot(range(1,len(rew_arr)+1),rew_arr)

# plt.subplot(1,2,2)
# plt.title("Total corrects per epoch")
# plt.plot(range(1,len(correct_arr)+1),correct_arr)

# plt.show()


import numpy as np
import pandas as pd

class EpsilonGreedyContextualBandit:

    def __init__(self, n_arms, n_contexts, epsilon):
        self.n_arms = n_arms
        self.n_contexts = n_contexts
        self.epsilon = epsilon
        self.q_values = np.zeros((n_arms, n_contexts))
        self.n_actions = np.zeros(n_arms)

    def select_arm(self, context):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_arms)
        else:
            return np.argmax(self.q_values[context])

    def update(self, context, action, reward):
        self.q_values[context, action] += (1 / self.n_actions[action]) * (reward - self.q_values[context, action])
        self.n_actions[action] += 1

def main():
    n_arms = 10
    n_contexts = 100
    epsilon = 0.1
    bandit = EpsilonGreedyContextualBandit(n_arms, n_contexts, epsilon)

    for i in range(10000):
        context = np.random.randint(n_contexts)
        action = bandit.select_arm(context)
        reward = np.random.randint(2)
        bandit.update(context, action, reward)

    print(bandit.q_values)

if __name__ == "__main__":
    main()