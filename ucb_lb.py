import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gym
from mmWave_bandits import mmWaveEnv
from matplotlib import pyplot as plt
import seaborn as sns



def main():


    env = mmWaveEnv()

    num_actions = (2, env.Nbeams) 

    A = [[np.random.rand(5, 5).astype(np.float32) for i in range(10)] for j in range(2)]
    b = [[np.random.rand(5, 1).astype(np.float32) for i in range(10)] for j in range(2)]

    Theta = [[0 for i in range(10)] for j in range(2)]

    for i in range(2):
        for j in range(10):
            Theta[i][j] = np.matmul(np.linalg.inv(A[i][j]), b[i][j])

    Q = np.array([np.array([0 for i in range(10)]) for j in range(2)])
    #########################################################


    num_epochs = 10 


    rew_arr = []
    correct_arr = []

    for epoch in range(num_epochs):
        A = [[np.random.rand(5, 5).astype(np.float32) for i in range(10)] for j in range(2)]
        b = [[np.random.rand(5, 1).astype(np.float32) for i in range(10)] for j in range(2)]

        Theta = [[0 for i in range(10)] for j in range(2)]

        for i in range(2):
            for j in range(10):
                Theta[i][j] = np.matmul(np.linalg.inv(A[i][j]), b[i][j])

        Q = np.array([np.array([0 for i in range(10)]) for j in range(2)])
        #########################################################

 

        max_time_steps = env.Horizon


        tot_corr = 0

        obs = env.reset()

        total_reward = 0

        for t in range(max_time_steps):


            if (t == 0):

                action = (np.random.randint(2), np.random.randint(env.Nbeams))

            else:
                # action = np.argmax(Q)
                for i in range(2):
                    for j in range(10):
                        product2 = np.matmul(np.matmul(trans_obs.T ,np.linalg.inv(A[i][j])) , trans_obs)[0][0]

                        product2_1 = ((0.2-0.01)*np.random.random_sample() + 0.01)*np.sqrt(product2)
                        
                        if product2 < 0:
                            product2_1 = ((0.2-0.01)*np.random.random_sample() + 0.01)*np.sqrt(-1*product2)

                        else:
                            product2_1 = ((0.2-0.01)*np.random.random_sample() + 0.01)*np.sqrt(product2)
                        Q[i][j] = np.matmul(trans_obs.T , Theta[i][j])[0][0] + product2_1

                action = np.unravel_index(np.argmax(Q, axis=None), Q.shape)

            new_obs, reward, correct, terminated, truncated, _ = env.step(action)

            if terminated:
                break
                

            tot_corr += correct
            total_reward += reward

            ########### Transformation part #############
            trans_obs = []

            for i in range(5):
                xt = new_obs[0][i]
                yt = new_obs[1][i]

                trans_obs.append(((xt-env.bs_location[0])**2 + (yt-env.bs_location[1])**2)**(0.5))

            trans_obs = np.array(trans_obs)
            trans_obs = trans_obs.reshape(5,1)
            ##############################################

            A[action[0]][action[1]] += np.matmul(trans_obs, trans_obs.T)
            b[action[0]][action[1]] += reward*trans_obs
            


            
        correct_arr.append(tot_corr)
        rew_arr.append(total_reward/max_time_steps)


        print()


        print(f"Epoch {epoch + 1}/{num_epochs}, Total Reward: {total_reward}")


    plt.figure(figsize=(20,7.5))
    plt.subplot(1,2,1)
    plt.title("lin_ucb: time averaged reward")
    plt.plot(range(1,len(rew_arr)+1),rew_arr)

    plt.subplot(1,2,2)
    plt.title("lin_ucb: time averaged accuracy")
    plt.plot(range(1,len(correct_arr)+1),correct_arr)

    plt.show()


if __name__ == "__main__":
    main()