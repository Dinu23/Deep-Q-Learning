import numpy as np

class ReplayBuffer():
    def __init__(self,size,mini_batch_size) -> None:
        self.size = size
        self._current_states = []
        self._actions = []
        self._rewards = []
        self._next_state = []
        self._terminated = []
        self._isFull = False
        self._position = 0
        self.mini_batch_size = mini_batch_size

    def add_experience(self,current_state, action, reward, next_state, terminated):
        
        if(not self._isFull):
            self._current_states.append(current_state)
            self._actions.append(action)
            self._rewards.append(reward)
            self._next_state.append(next_state)
            self._terminated.append(terminated)
            
        else:
            self._current_states[self._position]= current_state
            self._actions[self._position]= action
            self._rewards[self._position]= reward
            self._next_state[self._position]= next_state
            self._terminated[self._position]= terminated
            
        self._position +=1
        if(self._position > self.size):
            self._isFull = True
            self._position = 0
        
        
        

    def get_mini_batch(self):
        if(not  self._isFull and self.mini_batch_size > self._position):
            return [],[],[],[],[]
        if(self._isFull):
            positions = np.random.choice(self.size,self.mini_batch_size,replace=False)
        else:
            positions = np.random.choice(self._position,self.mini_batch_size, replace=False)
        
        current_states = np.array(self._current_states)[positions]
        actions = np.array(self._actions)[positions]
        rewards = np.array(self._rewards)[positions]
        next_states = np.array(self._next_state)[positions]
        terminated = np.array(self._terminated)[positions]

        return current_states,actions,rewards,next_states,terminated