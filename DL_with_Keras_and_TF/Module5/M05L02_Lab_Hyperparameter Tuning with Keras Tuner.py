#!/usr/bin/env python
# coding: utf-8

# <p style="text-align:center">
#     <a href="https://skills.network" target="_blank">
#     <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/assets/logos/SN_web_lightmode.png" width="200" alt="Skills Network Logo"  />
#     </a>
# </p>
# 

# # **Lab: Hyperparameter Tuning with Keras Tuner**
# 

# Estimated time needed: **30** minutes
# 

# In this lab, you will learn how to set up Keras Tuner and prepare the environment for hyperparameter tuning. 
# 
# ## Learning objectives: 
# By the end of this lab, you will: 
# - Install Keras Tuner and import the necessary libraries
# - Load and preprocess the MNIST data set
# - Define a model-building function that uses hyperparameters to configure the model architecture
# - Set up Keras Tuner to search for the best hyperparameter configuration 
# - Retrieve the best hyperparameters from the search and build a model with these optimized values
# 
# ## Prerequisites: 
# - Basic understanding of Python programming 
# - Keras and TensorFlow installed
# 

# ### Exercise 1: Install the Keras Tuner 
# 
# This exercise guides you through the initial setup for using Keras Tuner. You install the library, import necessary modules, and load and preprocess the MNIST data set, which will be used for hyperparameter tuning. 
# 1. **Install Keras Tuner:**
#     - Use pip to install Keras Tuner
# 2. **Import necessary libraries:**
#     - Import Keras Tuner, TensorFlow, and Keras modules
# 3. **Load and preprocess the MNIST data set:**
#     - Load the MNIST data set.
#     - Normalize the data set by dividing by 255.0.
# 

# In[1]:


#get_ipython().system('pip install tensorflow==2.16.2')
#get_ipython().system('pip install keras-tuner==1.4.7')
#get_ipython().system('pip install numpy<2.0.0')



# #### Explanation: 
# This code installs the necessary libraries using pip
# 
# - **TensorFlow**: Ensures compatibility with the Keras Tuner.
# - **Keras Tuner**: The version used in this lab.
# - **Numpy**: Ensures compatibility with the other installed packages.
# 

# In[2]:


import sys

# Increase recursion limit to prevent potential issues
sys.setrecursionlimit(100000)


# #### Explanation: 
# The sys.setrecursionlimit function is used to increase the recursion limit, which helps prevent potential recursion errors when running complex models with deep nested functions or when using certain libraries like TensorFlow.
# 

# In[3]:


# Step 2: Import necessary libraries
import keras_tuner as kt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam
import os
import warnings

# Suppress all Python warnings
warnings.filterwarnings('ignore')

# Set TensorFlow log level to suppress warnings and info messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = all logs, 1 = filter out INFO, 2 = filter out INFO and WARNING, 3 = ERROR only



# #### Explanation
# 
# This code imports the necessary libraries:
# 
# - **`keras_tuner`**: Used for hyperparameter tuning.
# - **`Sequential`**: A linear stack of layers in Keras.
# - **`Dense`**, **`Flatten`**: Common Keras layers.
# - **`mnist`**: The MNIST dataset, a standard dataset for image classification.
# - **`Adam`**: An optimizer in Keras.
# 

# In[4]:


# Step 3: Load and preprocess the MNIST dataset
(x_train, y_train), (x_val, y_val) = mnist.load_data()
x_train, x_val = x_train / 255.0, x_val / 255.0

print(f'Training data shape: {x_train.shape}')
print(f'Validation data shape: {x_val.shape}')


# #### Explanation
# 
# This code loads the MNIST dataset and preprocesses it:
# 
# - **`mnist.load_data()`**: Loads the dataset, returning training and validation splits.
# - **`x_train / 255.0`**: Normalizes the pixel values to be between 0 and 1.
# - **`print(f'...')`**: Displays the shapes of the training and validation datasets.
# 

# ### Exercise 2: Defining the model with hyperparameters 
# 
# In this exercise, you define a model-building function that uses the `HyperParameters` object to specify the number of units in a dense layer and the learning rate. This function returns a compiled Keras model that is ready for hyperparameter tuning.
# 
# **Define a model-building function:**
# - Create a function `build_model` that takes a `HyperParameters` object as input.
# - Use the `HyperParameters` object to define the number of units in a dense layer and the learning rate for the optimizer.
# - Compile the model with sparse categorical cross-entropy loss and Adam optimizer.
# 

# In[5]:


# Define a model-building function 

def build_model(hp):
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(units=hp.Int('units', min_value=32, max_value=512, step=32), activation='relu'),
        Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# #### Explanation
# 
# This function builds and compiles a Keras model with hyperparameters:
# 
# - **`hp.Int('units', ...)`**: Defines the number of units in the Dense layer as a hyperparameter.
# - **`hp.Float('learning_rate', ...)`**: Defines the learning rate as a hyperparameter.
# - **`model.compile()`**: Compiles the model with the Adam optimizer and sparse categorical cross-entropy loss.
# 

# ### Exercise 3: Configuring the hyperparameter search 
# 
# This exercise guides you through configuring Keras Tuner. You create a `RandomSearch` tuner, specifying the model-building function, the optimization objective, the number of trials, and the directory for storing results. The search space summary provides an overview of the hyperparameters being tuned. 
# 
# **Create a RandomSearch Tuner:**
# - Use the `RandomSearch` class from Keras Tuner. 
# - Specify the model-building function, optimization objective (validation accuracy), number of trials, and directory for storing results.
# 

# In[6]:


# Create a RandomSearch Tuner 

tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=2,
    directory='my_dir',
    project_name='intro_to_kt'
)

# Display a summary of the search space 
tuner.search_space_summary()


# #### Explanation
# 
# This code sets up a Keras Tuner `RandomSearch`:
# 
# - **`build_model`**: The model-building function.
# - **`objective='val_accuracy'`**: The metric to optimize (validation accuracy).
# - **`max_trials=10`**: The maximum number of different hyperparameter configurations to try.
# - **`executions_per_trial=2`**: The number of times to run each configuration.
# - **`directory='my_dir'`**: Directory to save the results.
# - **`project_name='intro_to_kt'`**: Name of the project for organizing results.
# 
# Displays a summary of the hyperparameter search space, providing an overview of the hyperparameters being tuned.
# 
# 

# ### Exercise 4: Running the hyperparameter search 
# 
# In this exercise, you run the hyperparameter search using the `search` method of the tuner. You provide the training and validation data along with the number of epochs. After the search is complete, the results summary displays the best hyperparameter configurations found. 
# 
# **Run the search:**
# - Use the `search` method of the tuner. 
# - Pass in the training data, validation data, and the number of epochs
# 

# In[8]:


# Run the hyperparameter search 
tuner.search(x_train, y_train, epochs=5, validation_data=(x_val, y_val)) 

# Display a summary of the results 
tuner.results_summary() 


# #### Explanation
# 
# This command runs the hyperparameter search:
# 
# - **`epochs=5`**: Each trial is trained for 5 epochs.
# - **`validation_data=(x_val, y_val)`**: The validation data to evaluate the model's performance during the search.
# 
# After the search is complete, this command displays a summary of the best hyperparameter configurations found during the search.
# 

# ## Exercise 5: Analyzing and using the best hyperparameters 
# 
# In this exercise, you retrieve the best hyperparameters found during the search and print their values. You then build a model with these optimized hyperparameters and train it on the full training data set. Finally, you evaluate the model’s performance on the test set to ensure that it performs well with the selected hyperparameters. 
# 
# **Retrieve the best hyperparameters:**
# - Use the `get_best_hyperparameters` method to get the best hyperparameters. 
# - Print the optimal values for the hyperparameters. 
# 
# **Build and train the model:**
# - Build a model using the best hyperparameters. 
# - Train the model on the full training data set and evaluate its performance on the test set.
# 

# In[9]:


# Step 1: Retrieve the best hyperparameters 

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0] 
print(f""" 

The optimal number of units in the first dense layer is {best_hps.get('units')}. 

The optimal learning rate for the optimizer is {best_hps.get('learning_rate')}. 

""") 

# Step 2: Build and Train the Model with Best Hyperparameters 
model = tuner.hypermodel.build(best_hps) 
model.fit(x_train, y_train, epochs=10, validation_split=0.2) 

# Evaluate the model on the test set 
test_loss, test_acc = model.evaluate(x_val, y_val) 
print(f'Test accuracy: {test_acc}') 


# #### Explanation
# 
# This code retrieves the best hyperparameters found during the search:
# 
# - **`get_best_hyperparameters(num_trials=1)`**: Gets the best hyperparameter configuration.
# - **`print(f"...")`**: Prints the best hyperparameters.
# - **`model.fit(...)`**: Trains the model on the full training data with a validation split of 20%.
# - **`model.evaluate(...)`**: Evaluates the model on the test (validation) dataset and prints the accuracy, which gives an indication of how well the model generalizes.
# 

# ## Practice exercises 
# 
# ### Exercise 1: Setting Up Keras Tuner 
# 
# #### Objective: 
# Learn how to set up Keras Tuner and prepare the environment for hyperparameter tuning. 
# 
# #### Instructions: 
# 1. Install Keras Tuner.
# 2. Import necessary libraries.
# 3. Load and preprocess the MNIST data set.
# 

# In[10]:


# Write your code here
#get_ipython().system('pip install keras-tuner')

# Step 2: Import necessary libraries 
import keras_tuner as kt 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Flatten 
from tensorflow.keras.datasets import mnist 
from tensorflow.keras.optimizers import Adam 

# Step 3: Load and preprocess the MNIST data set 
(x_train, y_train), (x_val, y_val) = mnist.load_data() 
x_train, x_val = x_train / 255.0, x_val / 255.0 

# Print the shapes of the training and validation datasets
print(f'Training data shape: {x_train.shape}') 
print(f'Validation data shape: {x_val.shape}')


# <details>
#     <summary>Click here for Solution</summary>
# 
# ```python
# !pip install keras-tuner 
# 
# # Step 2: Import necessary libraries 
# import keras_tuner as kt 
# from tensorflow.keras.models import Sequential 
# from tensorflow.keras.layers import Dense, Flatten 
# from tensorflow.keras.datasets import mnist 
# from tensorflow.keras.optimizers import Adam 
# 
# # Step 3: Load and preprocess the MNIST data set 
# (x_train, y_train), (x_val, y_val) = mnist.load_data() 
# x_train, x_val = x_train / 255.0, x_val / 255.0 
# 
# # Print the shapes of the training and validation datasets
# print(f'Training data shape: {x_train.shape}') 
# print(f'Validation data shape: {x_val.shape}')
# ```
# 
# </details>
# 

# ### Exercise 2: Defining the model with hyperparameters 
# 
# #### Objective: 
# Define a model-building function that uses hyperparameters to configure the model architecture. 
# 
# #### Instructions: 
# 1. Define a model-building function that uses the `HyperParameters` object to specify the number of units in a dense layer and the learning rate. 
# 2. Compile the model with sparse categorical cross-entropy loss and Adam optimizer. 
# 

# In[11]:


# Write your code here
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
import keras_tuner as kt

# Step 1: Define a model-building function
def build_model(hp):
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(units=hp.Int('units', min_value=32, max_value=512, step=32), activation='relu'),
        Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


# <details>
#     <summary>Click here for Solution</summary>
# 
# ```python
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Flatten
# from tensorflow.keras.optimizers import Adam
# import keras_tuner as kt
# 
# # Step 1: Define a model-building function
# def build_model(hp):
#     model = Sequential([
#         Flatten(input_shape=(28, 28)),
#         Dense(units=hp.Int('units', min_value=32, max_value=512, step=32), activation='relu'),
#         Dense(10, activation='softmax')
#     ])
# 
#     model.compile(
#         optimizer=Adam(learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')),
#         loss='sparse_categorical_crossentropy',
#         metrics=['accuracy']
#     )
#     
#     return model
# ```
# 
# </details>
# 

# ### Exercise 3: Configuring the hyperparameter search 
# 
# #### Objective: 
# Set up Keras Tuner to search for the best hyperparameter configuration. 
# 
# #### Instructions: 
# 1. Create a `RandomSearch` tuner using the model-building function. 
# 2. Specify the optimization objective, number of trials, and directory for storing results.
# 

# In[12]:


# Write your code here
import keras_tuner as kt

# Step 1: Create a RandomSearch Tuner
tuner = kt.RandomSearch(
    build_model,  # Ensure 'build_model' function is defined from previous code
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=2,
    directory='my_dir',
    project_name='intro_to_kt'
)

# Display a summary of the search space
tuner.search_space_summary()


# <details>
#     <summary>Click here for Solution</summary>
# 
# ```python
# import keras_tuner as kt
# 
# # Step 1: Create a RandomSearch Tuner
# tuner = kt.RandomSearch(
#     build_model,  # Ensure 'build_model' function is defined from previous code
#     objective='val_accuracy',
#     max_trials=10,
#     executions_per_trial=2,
#     directory='my_dir',
#     project_name='intro_to_kt'
# )
# 
# # Display a summary of the search space
# tuner.search_space_summary()
# ```
# 
# </details>
# 

# ### Exercise 4: Running the hyperparameter search
# 
# #### Objective: 
# Run the hyperparameter search and dispaly the summary of the results.
# 
# #### Instructions: 
# 1. Run the hyperparameter search using the `search` method of the tuner. 
# 2. Pass in the training data, validation data, and the number of epochs. 
# 3. Display a summary of the results. 
# 

# In[14]:


# Write your code here
# Step 1: Run the hyperparameter search 

tuner.search(x_train, y_train, epochs=5, validation_data=(x_val, y_val)) 

 # Display a summary of the results 

tuner.results_summary()


# <details>
#     <summary>Click here for Solution</summary>
# 
# ```python
# # Step 1: Run the hyperparameter search 
# 
# tuner.search(x_train, y_train, epochs=5, validation_data=(x_val, y_val)) 
# 
#  # Display a summary of the results 
# 
# tuner.results_summary()
# ```
# 
# </details>
# 

# ### Exercise 5: Analyzing and using the best hyperparameters 
# 
# #### Objective: 
# Retrieve the best hyperparameters from the search and build a model with these optimized values. 
# 
# #### Instructions: 
# 1. Retrieve the best hyperparameters using the `get_best_hyperparameters` method. 
# 2. Build a model using the best hyperparameters. 
# 3. Train the model on the full training data set and evaluate its performance on the validation set.
# 

# In[ ]:


# Write your code here
# Step 1: Retrieve the best hyperparameters 

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0] 

print(f""" 

The optimal number of units in the first dense layer is {best_hps.get('units')}. 

The optimal learning rate for the optimizer is {best_hps.get('learning_rate')}. 

""") 

 # Step 2: Build and train the model with best hyperparameters 

model = tuner.hypermodel.build(best_hps) 

model.fit(x_train, y_train, epochs=10, validation_split=0.2) 

 # Evaluate the model on the validation set 

val_loss, val_acc = model.evaluate(x_val, y_val) 

print(f'Validation accuracy: {val_acc}')


# <details>
#     <summary>Click here for Solution</summary>
# 
# ```python
# # Step 1: Retrieve the best hyperparameters 
# 
# best_hps = tuner.get_best_hyperparameters(num_trials=1)[0] 
# 
# print(f""" 
# 
# The optimal number of units in the first dense layer is {best_hps.get('units')}. 
# 
# The optimal learning rate for the optimizer is {best_hps.get('learning_rate')}. 
# 
# """) 
# 
#  # Step 2: Build and train the model with best hyperparameters 
# 
# model = tuner.hypermodel.build(best_hps) 
# 
# model.fit(x_train, y_train, epochs=10, validation_split=0.2) 
# 
#  # Evaluate the model on the validation set 
# 
# val_loss, val_acc = model.evaluate(x_val, y_val) 
# 
# print(f'Validation accuracy: {val_acc}') 
# ```
# 
# </details>
# 

# ### Conclusion 
# 
# Congratulations on completing this lab! You have learned to set up Keras Tuner and prepare the environment for hyperparameter tuning. In addition, you defined a model-building function that uses hyperparameters to configure the model architecture. You configured Keras Tuner to search for the best hyperparameter configuration and learned to run the hyperparameter search and analyze the results. Finally, you retrieved the best hyperparameters and built a model with these optimized values. 
# 

# ## Authors
# 

# Skillup
# 

# Copyright ©IBM Corporation. All rights reserved.
# 
