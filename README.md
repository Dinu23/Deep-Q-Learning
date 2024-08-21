
# Deep Q-Learning with Dueling and Double DQN Enhancements

## Overview

This repository implements advanced variants of the Deep Q-Learning algorithm, including Double DQN and Dueling DQN, which improve the stability and performance of the original DQN algorithm. These methods are applied to various reinforcement learning tasks in environments like OpenAI's Gym (we used CartPole-v1), demonstrating their effectiveness in reducing overestimation bias and enhancing the learning process.

## Features

- **Deep Q-Network (DQN)**: A baseline implementation of the DQN algorithm, which uses a neural network to approximate the Q-value function.
- **Experience Replay**: Stores agent experiences in a buffer to be sampled randomly during training, helping to break the correlation between consecutive experiences.
- **Target Network**: Utilizes a secondary network that is periodically updated to provide more stable targets during training.
- **Double DQN**: An extension of DQN that mitigates the overestimation bias by using a separate action selection process during the target value calculation.
- **Dueling DQN**: Introduces a dueling architecture that separately estimates the state value function and the advantage function, allowing the model to better differentiate between valuable and non-valuable actions in certain states.


## Key Components

### 1. Deep Q-Network (DQN)
- Implements the classic DQN algorithm that uses a neural network to approximate the Q-value function for each state-action pair.

### 2. Double DQN
- **Purpose**: Addresses the overestimation bias inherent in the original DQN by decoupling the action selection and target value calculation processes.
- **How it works**: Uses the primary network to select the best action and the target network to evaluate the Q-value of that action.

### 3. Dueling DQN
- **Purpose**: Improves the learning efficiency by estimating the value of being in a state separately from the advantage of each action in that state.
- **How it works**: Uses two streams within the neural network architecture – one to estimate the state value and another to estimate the advantage function.

## Getting Started

### Prerequisites

Ensure you have Python 3.x installed along with the necessary libraries. Install the dependencies using:

```bash
pip install -r requirements.txt
```

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Dinu23/Deep-Q-Learning.git
   ```
2. Navigate to the project directory:
   ```bash
   cd Deep-Q-Learning
   ```

### Running the Algorithms

To train the DQN with a specific variant on a given environment, you can use the following commands:

- **Standard DQN**:
  ```bash
  python DQN.py
  ```
- **Double DQN**:
  ```bash
  python DoubleDQN.py
  ```
- **Dueling DQN**:
  ```bash
  python DuelingDQN.py
  ```

## Contact

For questions, issues, or any other inquiries, please reach out to:

- **Name**: Dinu Catalin-Viorel
- **Email**: viorel.dinu00@gmail.com

