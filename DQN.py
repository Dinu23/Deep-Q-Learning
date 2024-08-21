import gym
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Input
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import argparse
import random
from policy import EGreedy, Boltzman, Noisy, SimHash
from model import N_Network,Dueling_Network
from Helper import argmax
from replay_buffer import ReplayBuffer

# gpus = tf.config.list_physical_devices("GPU")
# tf.config.set_visible_devices(gpus[0], "GPU")
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def custom_loss_function(y_true, y_pred):
   squared_difference = tf.square(y_true - y_pred)
   return tf.reduce_sum(squared_difference, axis=-1)


class DQNAgent:
    def __init__(self, env, model_builder = N_Network(),k =None):
        self.env = env
        self.model_builder = model_builder
        self.state_shape = env.observation_space.shape[0]
        self.n_actions = env.action_space.n
        self.model = model_builder.create_model(self.state_shape, self.n_actions)
        if(k != None):
            self.hasher = SimHash( env.observation_space.shape[0],k) 

    def train_dqn(self,
                no_of_episodes=5,
                optimizer=Adam(learning_rate=0.001),
                gamma=1,
                policy=EGreedy(0.3),
                TN=True,
                freq_update_target_network=1000,
                RB=True,
                freq_update_network_from_experience=5,
                replay_buffer=ReplayBuffer(10000, 32),
                DDQN = False,
                count_based = False,
                beta = 1,
                verbose = False
                ):
        
        self.model.compile(optimizer=optimizer,loss= custom_loss_function)

        if TN:
            target_model = self.model_builder.create_model(self.state_shape, self.n_actions)
            target_model.set_weights(self.model.get_weights())

        step = 0
        rewards = []

        for episode in range(no_of_episodes):
            with tf.GradientTape() as tape:
                current_state,info = self.env.reset()
                episode_reward = 0
                loss = 0
                while True:
                    Q_sa = self.model(np.array([current_state]))
                    # print(np.array([current_state]))
                    step += 1
                    action = policy.select_action(Q_sa[0],step,no_of_episodes)
                    
                    next_state, reward, terminated, truncated,info  = self.env.step(action)

                    if terminated:
                        reward = 0
                       

                    if not RB:
                        output = self.model(np.array([current_state]))[0][action]
                        
                        if TN:
                            target = reward + gamma * np.max(target_model(np.array([next_state])))
                        else:
                            target = reward + gamma * np.max(self.model(np.array([next_state])))

                        if terminated:
                            target = reward
                    
                        loss += (target - output) ** 2

                    if RB:
                        replay_buffer.add_experience(current_state, action, reward, next_state, 0 if terminated else 1)

                        if step % freq_update_network_from_experience == 0:
                            current_states,actions,state_rewards,next_states,terminated_states = replay_buffer.get_mini_batch()

                            if len(current_states) > 0:
                                if(count_based):
                                    counts = self.hasher.count(current_states)
                                    state_rewards += beta/np.sqrt(counts)
                                    # print(state_rewards)


                                target = np.array(self.model(current_states))
                                index = np.arange(replay_buffer.mini_batch_size, dtype=np.int32)
                                if TN:
                                    if(DDQN):
                                        best_action_next_state = np.argmax(self.model(next_states), axis =1)
                                        target[index,actions] = state_rewards + terminated_states * gamma * np.array(target_model(next_states))[index,best_action_next_state]
                                    else:
                                        target[index,actions] = state_rewards + terminated_states * gamma * np.max(target_model(next_states),axis =1)
                                else:
                                    target[index,actions] = state_rewards + terminated_states * gamma * np.max(self.model(next_states),axis =1)   
                                # print(target)
                                self.model.train_on_batch(current_states,target,return_dict=True)



                    if TN and step % freq_update_target_network == 0:
                        target_model.set_weights(self.model.get_weights())

                    current_state = next_state
                    episode_reward += reward

                    if terminated or truncated:
                        self.env.reset()
                        break

                rewards.append(episode_reward)
                
                if(verbose and episode %50  ==0):
                    print(f'Episode {episode + 1} -> Reward: {episode_reward}')

                if not RB:
                    gradients = tape.gradient(loss, self.model.trainable_variables)
                    optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return rewards

    
def experiment(no_of_episodes, target_network, experience_replay):
    # no_of_episodes = 10
    model_builder = N_Network([512])
    env = gym.make("CartPole-v1")

    agent = DQNAgent(env,model_builder,15)



    learning_rate = 0.01
    optimizer = Adam(learning_rate=learning_rate)
    gamma = 1
    policy = EGreedy(0.99,annealing=True, epsilon_end = 0.01, percentage= 0.5)
    TN = target_network
    freq_update_target_network = 1000
    RB = experience_replay
    freq_update_network_from_experience = 5
    buffer_length, batch_size = 10000, 32
    DDQN = False,
    count_based= True
    beta = 1
    verbose = True
    rewards = agent.train_dqn(
        no_of_episodes=no_of_episodes,
        optimizer=optimizer,
        gamma=gamma,
        policy=policy,
        TN=TN,
        freq_update_target_network=freq_update_target_network,
        RB=RB,
        freq_update_network_from_experience=freq_update_network_from_experience,
        replay_buffer=ReplayBuffer(buffer_length, batch_size),
        DDQN = DDQN,
        count_based = count_based,
        beta = beta,
        verbose = verbose
    )

    print(f"Rewards we got for {no_of_episodes} episodes are : {rewards}")
   

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deep Q-Network (DQN) - CartPole V1')
    parser.add_argument('--no-TN', action='store_false', help='Disable target network')
    parser.add_argument('--no-ER', action='store_false', help='Disable experience replay')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes to run')
    args = parser.parse_args()

    experiment(args.episodes, args.no_TN, args.no_ER)