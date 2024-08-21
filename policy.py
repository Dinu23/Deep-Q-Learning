import numpy as np
from Helper import softmax
class Policy:
    def __init__(self) -> None:
        pass

    def select_action(self, Q_sa, t, t_max):
        pass


class EGreedy(Policy):
    def __init__(self, epsilon_start, annealing = False, epsilon_end=None, percentage=None ) -> None:
        super().__init__()
        self.epsilon_start = epsilon_start
        self.annealing = annealing
        self.epsilon_end = epsilon_end
        self.percentage = percentage
    

    def select_action(self, Q_sa, t, t_max):
        if(self.annealing):
            final_from_T = int(self.percentage * t_max)
            if t > final_from_T:
                epsilon = self.epsilon_end
            else:
                epsilon =  self.epsilon_end + (self.epsilon_start - self.epsilon_end) * (final_from_T - t)/final_from_T
        else:
            epsilon = self.epsilon_start

        best_action = np.argmax(Q_sa)
        if np.random.random() > epsilon:
            return best_action
        else:
            return np.random.randint(len(Q_sa))
        

class Boltzman(Policy):
    def __init__(self, temp_start, annealing = False, temp_end=None, percentage=None ) -> None:
        super().__init__()
        self.temp_start = temp_start
        self.annealing = annealing
        self.temp_end = temp_end
        self.percentage = percentage
    

    def select_action(self, Q_sa, t, t_max):
        if(self.annealing):
            final_from_T = int(self.percentage * t_max)
            if t > final_from_T:
                temp = self.temp_end
            else:
                temp =  self.temp_end + (self.temp_start - self.temp_end) * (final_from_T - t)/final_from_T
        else:
            temp = self.temp_start  
        
        probabilities = softmax(Q_sa,temp)
           
        return np.random.choice(len(Q_sa), p = probabilities) 

class Noisy(Policy):
    def __init__(self, scale_start, scale_end=None, percentage=None ) -> None:
        super().__init__()
        self.scale_start = scale_start
        self.scale_end = scale_end
        self.percentage = percentage

    def select_action(self, Q_sa, t, t_max):
        
        final_from_T = int(self.percentage * t_max)
        if t > final_from_T:
            scale = self.scale_end
        else:
            scale =  self.scale_end + (self.scale_start - self.scale_end) * (final_from_T - t)/final_from_T
   
        
        a_noise = np.random.normal(0, scale, size=self.n_actions)
        noisy_a_values = Q_sa + a_noise
        noisy_action = np.argmax(noisy_a_values)
           
        return noisy_action
    
class SimHash:
    def __init__(self,state_size,k) -> None:
        self.hash_states = {}
        self.A = np.random.normal(0,1, (k , state_size))

    def count(self,states):
        counts = []
        for state in states :
            state_hash = str(np.sign(self.A @ state))
            # print(state_hash)
            if state_hash in self.hash_states:
                self.hash_states[state_hash] = self.hash_states[state_hash] + 1
            else :
                self.hash_states[state_hash] = 1
            counts.append(self.hash_states[state_hash])
        # print(self.hash_states)
        return np.array(counts)