import gym
import numpy as np
import tensorflow as tf
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import argparse
from policy import EGreedy, Boltzman
from model import N_Network,Dueling_Network
from replay_buffer import ReplayBuffer
import os

# gpus = tf.config.list_physical_devices("GPU")
# tf.config.set_visible_devices(gpus[0], "GPU")
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def custom_loss_function(y_true, y_pred):
   squared_difference = tf.square(y_true - y_pred)
   return tf.reduce_sum(squared_difference, axis=-1)


class DuelingDQNAgent:
    def __init__(self, env, model_builder = Dueling_Network()):
        self.env = env
        self.model_builder = model_builder
        self.state_shape = env.observation_space.shape[0]
        self.n_actions = env.action_space.n
        self.model, self.model_advantage = model_builder.create_model(self.state_shape, self.n_actions)


    def train_dqn(self,
                no_of_episodes=5,
                optimizer=Adam(learning_rate=0.001),
                gamma=1,
                policy=EGreedy(0.3),
                freq_update_target_network=1000,
                freq_update_network_from_experience=5,
                replay_buffer=ReplayBuffer(10000, 32),
                verbose = False
                ):
        
        self.model.compile(optimizer=optimizer,loss= custom_loss_function)
        target_model, _ = self.model_builder.create_model(self.state_shape, self.n_actions)
        target_model.set_weights(self.model.get_weights())

        step = 0
        rewards = []

        for episode in range(no_of_episodes):
            current_state,info = self.env.reset()
            count = 0
            while True:
                Q_sa_advantage = self.model_advantage(np.array([current_state]))
                # print(np.array([current_state]))
                step += 1
                action = policy.select_action(Q_sa_advantage[0],step,no_of_episodes)
                next_state, reward, terminated, truncated, info = self.env.step(action)

                if terminated:
                    reward = 0
                    

                replay_buffer.add_experience(current_state, action, reward, next_state, 0 if terminated else 1)

                if step % freq_update_network_from_experience == 0:
                    current_states,actions,state_rewards,next_states,terminated_states = replay_buffer.get_mini_batch()

                    if len(current_states) > 0:
                        target = np.array(self.model(current_states))
                        index = np.arange(replay_buffer.mini_batch_size, dtype=np.int32)
                        target[index,actions] = state_rewards + terminated_states * gamma * np.max(target_model(next_states),axis =1)
                       
                        self.model.train_on_batch(current_states,target,return_dict=True)
                if step % freq_update_target_network == 0:
                    target_model.set_weights(self.model.get_weights())

                current_state = next_state
                count += 1

                if terminated or truncated:
                    self.env.reset()
                    break

            rewards.append(count)
            
            if(verbose and episode %50  ==0):
                print(f'Episode {episode + 1} -> Reward: {count}')


        return rewards

def moving_average(data, window_size):
    cumsum = np.cumsum(np.insert(data, 0, 0))
    return (cumsum[window_size:] - cumsum[:-window_size]) / window_size

def plot_rewards(rewards, window_size=10):
    smoothed_rewards = moving_average(rewards, window_size)

    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label='Rewards')
    plt.plot(np.arange(window_size - 1, len(rewards)), smoothed_rewards, label=f'Smoothed rewards (window = {window_size})')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Rewards over episodes')
    plt.legend()
    plt.show()
    # plt.savefig('dqn_rewards.png')
    
def experiment(no_of_episodes):
    # no_of_episodes = 10
    model_builder = Dueling_Network([512,256,256])
    env = gym.make("CartPole-v1")

    agent = DuelingDQNAgent(env,model_builder)



    learning_rate = 0.01
    optimizer = Adam(learning_rate=learning_rate)
    gamma = 1
    policy = EGreedy(0.99,annealing=True, epsilon_end = 0.01, percentage= 0.5)
    freq_update_target_network = 1000
    freq_update_network_from_experience = 5
    buffer_length, batch_size = 10000, 32
    verbose = True

    rewards = agent.train_dqn(
        no_of_episodes=no_of_episodes,
        optimizer=optimizer,
        gamma=gamma,
        policy=policy,
        freq_update_target_network=freq_update_target_network,
        freq_update_network_from_experience=freq_update_network_from_experience,
        replay_buffer=ReplayBuffer(buffer_length, batch_size),
        verbose = verbose
    )

    print(f"Rewards we got for {no_of_episodes} episodes are : {rewards}")
    plot_rewards(rewards, window_size=10)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deep Q-Network (DQN) - CartPole V1')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes to run')
    args = parser.parse_args()

    experiment(args.episodes)