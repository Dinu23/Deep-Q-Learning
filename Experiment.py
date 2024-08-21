#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
import time
from replay_buffer import ReplayBuffer
from keras.optimizers import Adam

from DQN_ import train_dqn
from Helper import LearningCurvePlot, smooth

def average_over_repetitions(n_repetitions, no_of_episodes, 
                            optimizer='Adam',learning_rate=0.01, gamma=1, policy='egreedy',
                            TN=True, RB=True, freq_update_target_network=1000,
                            freq_update_network_from_experience = 5, 
                            buffer_length=100000, batch_size=32, epsilon=None, 
                            temp=None, smoothing_window=51, plot=False):

    reward_results = np.empty([n_repetitions,no_of_episodes]) # Result array
    now = time.time()
    
    optimizer_ = Adam(learning_rate=learning_rate)
        
    for rep in range(n_repetitions): # Loop over repetitions
        # rewards = train_dqn(no_of_episodes=no_of_episodes,
        #             optimizer=optimizer_,
        #             gamma=gamma,git 
        #             policy=policy,
        #             TN=TN,
        #             freq_update_target_network=freq_update_target_network,
        #             RB=RB,
        #             freq_update_network_from_experience=freq_update_network_from_experience,
        #             replay_buffer=ReplayBuffer(buffer_length, batch_size))
        rewards = train_dqn(no_of_episodes = 100,
               optimizer = Adam(learning_rate=0.001),
               gamma = 1,
               policy = 'egreedy',
               TN = True,
               freq_update_target_network = 1000,
               RB = True,
               freq_update_network_from_experience = 5,
               replay_buffer = ReplayBuffer(10000,32))
        reward_results[rep] = rewards
        
    print('Running one setting takes {} minutes'.format((time.time()-now)/60))    
    learning_curve = np.mean(reward_results,axis=0) # average over repetitions
    learning_curve = smooth(learning_curve,smoothing_window) # additional smoothing
    return learning_curve  

def experiment():
    ####### Settings
    # Experiment    
    n_repetitions = 2
    smoothing_window = 5
    plot = False # Plotting is very slow, switch it off when we run repetitions
    
    # MDP    
    # n_timesteps = 50000
    gamma = 1.0
    epsilon, temp = None, None
    no_of_episodes = 100
    learning_rate = 0.01
    optimizer = 'Adam',
    gamma = 1,
    policy = 'egreedy',
    TN = True,
    freq_update_target_network = 1000,
    RB = True,
    freq_update_network_from_experience = 5,
    buffer_length, batch_size = 10000, 32
    
    Plot = LearningCurvePlot(title = 'Exploration: $\epsilon$-greedy versus softmax exploration')    
        
    learning_curve = average_over_repetitions(n_repetitions, no_of_episodes, 
                            optimizer,learning_rate, gamma, policy,
                            TN, RB, freq_update_target_network,
                            freq_update_network_from_experience, 
                            buffer_length, batch_size, epsilon, 
                            temp, smoothing_window, plot)
    Plot.add_curve(learning_curve,label=r'$\epsilon$-greedy')
    Plot.save('exploration.png')
    
if __name__ == '__main__':
    experiment()
