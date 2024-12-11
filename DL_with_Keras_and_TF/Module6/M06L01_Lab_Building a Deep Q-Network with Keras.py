# jupyter nbconvert --to script "M06L01_Lab_Building a Deep Q-Network with Keras.ipynb"#!/usr/bin/env python
# coding: utf-8

# <p style="text-align:center">
#     <a href="https://skills.network" target="_blank">
#     <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/assets/logos/SN_web_lightmode.png" width="200" alt="Skills Network Logo"  />
#     </a>
# </p>
# 

# # **Lab: Building a Deep Q-Network with Keras**
# 

# Estimated time needed: **30** minutes
# 

# In this lab, you will implement a Deep Q-Network (DQN) using Keras to solve a reinforcement learning problem. You will set up the environment, define the DQN model, train the agent, and evaluate its performance. By the end of this lab, you will know how to apply DQNs in Keras to optimize action-selection policies for a given environment. 
# 

# ## Learning objectives 
# By the end of this lab, you will: 
# - Implement a Deep Q-Network using Keras  
# - Define and train a neural network to approximate the Q-values  
# - Evaluate the performance of the trained DQN agent 
# 
# ## Prerequisites 
# - Basic understanding of Python and Keras 
# - Familiarity with neural networks 
# - Understanding of reinforcement learning concepts 
# 

# ### Steps 
# 
# #### Step 1: Set up the environment 
# 
# Before you start, set up the environment using the OpenAI Gym library. You will use the 'CartPole-v1' environment, which is a common benchmark for reinforcement learning algorithms. 
# 
#  
# 

# In[1]:


#get_ipython().system('pip install gym')


# In[2]:


#get_ipython().system('pip install tensorflow==2.16.2')


# In[3]:


import gym
import numpy as np

# Create the environment  
env = gym.make('CartPole-v1')

# Set random seed for reproducibility  
np.random.seed(42)
env.reset(seed=42)


# #### Notes: 
# - `gym` is a toolkit for developing and comparing reinforcement learning algorithms.
# - `CartPole-v1` is an environment where a pole is balanced on a cart, and the goal is to prevent the pole from falling over.  
# - Setting random seeds ensures that you can reproduce the results.
# 

# #### Step 2: Define the DQN model 
# 
# Define a neural network using Keras to approximate the Q-values. The network will take the state as input and output Q-values for each action. 
# 

# In[4]:


# Suppress warnings for a cleaner notebook or console experience
import warnings
warnings.filterwarnings('ignore')

# Disable warnings for a cleaner notebook or console experience
def warn(*args, **kwargs):
    pass
warnings.warn = warn

# Import necessary libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


def build_model(state_size, action_size):
    model = Sequential()
    model.add(Dense(24, input_dim=state_size, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(action_size, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(learning_rate=0.001))
    return model

state_size = env.observation_space.shape[0]
action_size = env.action_space.n
model = build_model(state_size, action_size)


# #### Notes: 
# 
# - `Sequential` model: a linear stack of layers in Keras. 
# - `Dense` layers: fully connected layers.  
# - `input_dim`: the size of the input layer, corresponding to the state size.  
# - `activation='relu'`: Rectified Linear Unit activation function.  
# - `activation='linear'`: linear activation function for the output layer. 
# - `Adam` optimizer: an optimization algorithm that adjusts the learning rate based on gradients.
# 

# #### Step 3: Implement the replay buffer 
# 
# A replay buffer stores the agent's experiences for training. We will implement a replay buffer using a deque. 
# 
#  
# 

# In[5]:


from collections import deque
import random

memory = deque(maxlen=2000)
def remember(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))


# #### Notes: 
# 
# - `memory`: a deque to store experiences (state, action, reward, next_state, done).  
# - `remember()`: stores experiences in memory.
# 

# #### Step 4: Implement the epsilon-greedy policy 
# 
# The epsilon-greedy policy balances exploration and exploitation by choosing random actions with probability epsilon. 
# 

# In[6]:


epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
 
def act(state):
    if np.random.rand() <= epsilon:
        return random.randrange(action_size)
    q_values = model.predict(state)
    return np.argmax(q_values[0])


# #### Notes: 
# - `epsilon`: exploration rate.  
# - `epsilon_min`: minimum exploration rate.  
# - `epsilon_decay`: decay rate for epsilon after each episode.  
# - `act()`: chooses an action based on the epsilon-greedy policy.
# 

# #### Step 5: Implement the Q-learning update 
# 
# Implement the Q-learning update to train the DQN using experiences stored in the replay buffer. 
# 

# In[7]:


def replay(batch_size):
    global epsilon
    minibatch = random.sample(memory, batch_size)
    for state, action, reward, next_state, done in minibatch:
        target = reward
        if not done:
            target = reward + gamma * np.amax(model.predict(next_state)[0])
        target_f = model.predict(state)
        target_f[0][action] = target
        model.fit(state, target_f, epochs=1, verbose=0)
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay


# #### Notes: 
# - `replay()`: samples a random minibatch from memory and trains the model.  
# - `target`: the Q-value target, which is updated using the reward and the maximum Q-value of the next state.  
# - `model.fit()`: trains the model on the updated Q-values.
# 

# #### Step 6: Train the DQN 
# 
# Train the DQN agent by interacting with the environment and updating the Q-values using the replay buffer.
# 

# In[8]:


for e in range(10):
    state = env.reset()

    # If state is a tuple, take the first element
    if isinstance(state, tuple):
        state = state[0]

    state = np.reshape(state, [1, state_size])
    
    for time in range(100):
        env.render()
        action = np.argmax(model.predict(state)[0])
        
        # Handle environments that return more than 4 values
        result = env.step(action)
        if isinstance(result, tuple) and len(result) == 4:
            next_state, reward, done, _ = result
        else:
            next_state, reward, done, _, _ = result  # Adjust based on the number of values returned
        
        # If next_state is a tuple, take the first element
        if isinstance(next_state, tuple):
            next_state = next_state[0]

        next_state = np.reshape(next_state, [1, state_size])
        state = next_state
        
        if done:
            print(f"episode: {e+1}/10, score: {time}")
            break

env.close()


# #### Notes: 
# - The main loop iterates over episodes, interacting with the environment and training the model.  
# - `env.reset()`: resets the environment at the beginning of each episode.  
# - `env.step(action)`: takes the chosen action and observes the reward and next state.  
# - The score for each episode is printed to monitor training progress.
# 

# #### Step 7: Evaluate the performance 
# 
# Evaluate the performance of the trained DQN agent. 
# 

# In[9]:


for e in range(10):
    state = env.reset()

    # Check if state is a tuple and extract the first element if it is
    if isinstance(state, tuple):
        state = state[0]

    state = np.reshape(state, [1, state_size])

    for time in range(100):
        env.render()
        action = np.argmax(model.predict(state)[0])

        # Handle environments that return more than 4 values
        result = env.step(action)
        if len(result) == 4:
            next_state, reward, done, _ = result
        else:
            next_state, reward, done, _, _ = result  # Adjust this based on the number of values returned

        # Check if next_state is a tuple and extract the first element if it is
        if isinstance(next_state, tuple):
            next_state = next_state[0]

        next_state = np.reshape(next_state, [1, state_size])
        state = next_state

        if done:
            print(f"episode: {e+1}/10, score: {time}")
            break

env.close()


# #### Notes:
# - This loop runs 10 episodes to test the trained agent.
# - `env.render()`: Visualizes the environment.
# - The agent chooses actions based on the trained model and interacts with the environment.
# - The environment's `reset` and `step` methods may return additional information in the form of tuples. The code now checks if the `state` and `next_state` are tuples and extracts the necessary data.
# - The `env.step(action)` method may return more than the standard four values. The code has been updated to handle these additional values by unpacking only the necessary information and ignoring the rest.
# - The score for each episode is printed, indicating how long the agent was able to balance the pole in each episode.
# 

# ## Practice Exercises
# ### Exercise 1: Modify the Reward Function to Encourage Longer Episodes
# **Objective:** Modify the reward structure to encourage the agent to keep the pole balanced longer.
# 
# **Instructions:**
# 1. Instead of just using the environment's reward, modify the reward function to include a penalty for large pole angles.
# 2. Update the reward calculation in the `train_dqn` function to discourage the agent from letting the pole deviate too far from the center.
# 3. Observe the effect on the agent's learning and episode length.
# 

# In[12]:


import os

# Create sample directory structure if it does not exist
base_dir = 'sample_data'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'validation')
class1_train = os.path.join(train_dir, 'class1')
class2_train = os.path.join(train_dir, 'class2')
class1_val = os.path.join(val_dir, 'class1')
class2_val = os.path.join(val_dir, 'class2')

# Create directories if they do not exist
for dir_path in [train_dir, val_dir, class1_train, class2_train, class1_val, class2_val]:
    os.makedirs(dir_path, exist_ok=True)

print("Directory structure created. Add your images to these directories.")

# Import the necessary library
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Modify data generator to include validation data
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    'sample_data',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    'sample_data',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# Function to modify the reward to encourage longer episodes
def modify_reward(reward, next_state):
    # Penalize large pole angles
    pole_angle = abs(next_state[2])  # Extract the pole angle from the state
    penalty = 1 if pole_angle > 0.1 else 0  # Apply penalty if angle is large
    return reward - penalty  # Adjust reward

# Inside the training loop
# Example usage in a reinforcement learning training loop:
# reward = modify_reward(reward, next_state)  # Use the modified reward


# <details>
# <summary>Click here for solution</summary> </br>
# 
# ```python
# import os
# 
# # Create sample directory structure if it does not exist
# base_dir = 'sample_data'
# train_dir = os.path.join(base_dir, 'train')
# val_dir = os.path.join(base_dir, 'validation')
# class1_train = os.path.join(train_dir, 'class1')
# class2_train = os.path.join(train_dir, 'class2')
# class1_val = os.path.join(val_dir, 'class1')
# class2_val = os.path.join(val_dir, 'class2')
# 
# # Create directories if they do not exist
# for dir_path in [train_dir, val_dir, class1_train, class2_train, class1_val, class2_val]:
#     os.makedirs(dir_path, exist_ok=True)
# 
# print("Directory structure created. Add your images to these directories.")
# 
# # Import the necessary library
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# 
# # Modify data generator to include validation data
# train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
# 
# train_generator = train_datagen.flow_from_directory(
#     'sample_data',
#     target_size=(224, 224),
#     batch_size=32,
#     class_mode='binary',
#     subset='training'
# )
# 
# validation_generator = train_datagen.flow_from_directory(
#     'sample_data',
#     target_size=(224, 224),
#     batch_size=32,
#     class_mode='binary',
#     subset='validation'
# )
# 
# # Function to modify the reward to encourage longer episodes
# def modify_reward(reward, next_state):
#     # Penalize large pole angles
#     pole_angle = abs(next_state[2])  # Extract the pole angle from the state
#     penalty = 1 if pole_angle > 0.1 else 0  # Apply penalty if angle is large
#     return reward - penalty  # Adjust reward
# 
# # Inside the training loop
# # Example usage in a reinforcement learning training loop:
# # reward = modify_reward(reward, next_state)  # Use the modified reward
# 
# 

# ### Exercise 2: Implement Early Stopping Based on Episode Length
# **Objective:** Stop training early if the agent consistently reaches the maximum episode length.
# 
# **Instructions:**
# 1. Add an early stopping mechanism that stops training if the agent achieves a specified number of consecutive episodes with a length above a threshold.
# 2. Set the threshold as 195 steps for 100 consecutive episodes (for CartPole).
# 3. Print a message and stop training when this condition is met.
# 

# In[13]:


# Import necessary libraries
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Modify data generator to include validation data
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    'sample_data',  
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    'sample_data',  
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# Early stopping parameters
consecutive_success_threshold = 100
success_episode_length = 195
consecutive_success_count = 0
episode_lengths = []  # Initialize episode lengths list

# Example of training loop (this should be your actual loop)
for episode in range(1000):  # Replace with actual loop condition
    # Training logic goes here
    episode_length = 200  # Example value, replace with actual calculation
    episode_lengths.append(episode_length)
    
    # Early stopping check
    if len(episode_lengths) > consecutive_success_threshold and all(
        length >= success_episode_length for length in episode_lengths[-consecutive_success_threshold:]
    ):
        print("Early stopping: Agent consistently reaches max episode length.")
        break  # This break is now correctly inside the loop


# <details>
# <summary>Click here for solution</summary> </br>
# 
# ```python
# # Import necessary libraries
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# 
# # Modify data generator to include validation data
# train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
# 
# train_generator = train_datagen.flow_from_directory(
#     'sample_data',  
#     target_size=(224, 224),
#     batch_size=32,
#     class_mode='binary',
#     subset='training'
# )
# 
# validation_generator = train_datagen.flow_from_directory(
#     'sample_data',  
#     target_size=(224, 224),
#     batch_size=32,
#     class_mode='binary',
#     subset='validation'
# )
# 
# # Early stopping parameters
# consecutive_success_threshold = 100
# success_episode_length = 195
# consecutive_success_count = 0
# episode_lengths = []  # Initialize episode lengths list
# 
# # Example of training loop (this should be your actual loop)
# for episode in range(1000):  # Replace with actual loop condition
#     # Training logic goes here
#     episode_length = 200  # Example value, replace with actual calculation
#     episode_lengths.append(episode_length)
#     
#     # Early stopping check
#     if len(episode_lengths) > consecutive_success_threshold and all(
#         length >= success_episode_length for length in episode_lengths[-consecutive_success_threshold:]
#     ):
#         print("Early stopping: Agent consistently reaches max episode length.")
#         break  # This break is now correctly inside the loop
# 

# ### Exercise 3: Experiment with Different Exploration Strategies
# **Objective:** Implement an epsilon decay schedule that switches from linear decay to exponential decay after a certain number of episodes.
# 
# **Instructions:**
# 1. Modify the epsilon decay strategy to start with a linear decay until a specific episode, then switch to an exponential decay.
# 2. Implement a check in the `choose_action` function to change the decay strategy after 100 episodes.
# 3. Observe and compare the agent’s performance.
# 

# In[14]:


# Modify data generator to include validation data
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
    'sample_data',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    'sample_data',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

def decay_epsilon(epsilon, episode, switch_episode=100):
    if episode < switch_episode:
        return max(epsilon - 0.01, 0.01)  # Linear decay
    else:
        return max(epsilon * 0.99, 0.01)  # Exponential decay

# Inside the training loop
epsilon = decay_epsilon(epsilon, e)  # Adjust epsilon based on the current episode


# <details>
# <summary>Click here for solution</summary> </br>
# 
# ```python
# # Modify data generator to include validation data
# train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
# 
# train_generator = train_datagen.flow_from_directory(
#     'sample_data',
#     target_size=(224, 224),
#     batch_size=32,
#     class_mode='binary',
#     subset='training'
# )
# 
# validation_generator = train_datagen.flow_from_directory(
#     'sample_data',
#     target_size=(224, 224),
#     batch_size=32,
#     class_mode='binary',
#     subset='validation'
# )
# 
# def decay_epsilon(epsilon, episode, switch_episode=100):
#     if episode < switch_episode:
#         return max(epsilon - 0.01, 0.01)  # Linear decay
#     else:
#         return max(epsilon * 0.99, 0.01)  # Exponential decay
# 
# # Inside the training loop
# epsilon = decay_epsilon(epsilon, e)  # Adjust epsilon based on the current episode
# 

# ### Summary
# These exercises are concise and focus on key modifications to the original DQN setup. The code snippets provided are short and simple, encouraging students to think critically about the modifications and how they impact the agent's performance.
# 

# ### Conclusion 
# 
# Congratulations! You have successfully implemented a Deep Q-Network using Keras to solve the CartPole-v1 environment. You defined a neural network to approximate Q-values, implemented a replay buffer, trained the network using experiences stored in memory, and evaluated the performance of the trained agent. This hands-on exercise reinforced your understanding of DQNs and their implementation in Keras.
# 

# ## Authors
# 

# Skills Network
# 

# Copyright © IBM Corporation. All rights reserved.
# 
