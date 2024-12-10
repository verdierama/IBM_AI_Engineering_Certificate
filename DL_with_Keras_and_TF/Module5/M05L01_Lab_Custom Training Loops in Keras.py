# jupyter nbconvert --to scripT "M05L01_Lab_Custom Training Loops in Keras.ipynb"
# #!/usr/bin/env python
# coding: utf-8

# <p style="text-align:center">
#     <a href="https://skills.network" target="_blank">
#     <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/assets/logos/SN_web_lightmode.png" width="200" alt="Skills Network Logo"  />
#     </a>
# </p>
# 

# # **Lab: Custom Training Loops in Keras**
# 

# Estimated time needed: **30** minutes
# 

# In this lab, you will learn to implement a basic custom training loop in Keras. 
# 

# ## Objectives
# 
# By the end of this lab, you will: 
# 
# - Set up the environment 
# 
# - Define the neural network model 
# 
# - Define the Loss Function and Optimizer 
# 
# - Implement the custom training loop 
# 
# - Enhance the custom training loop by adding an accuracy metric to monitor model performance 
# 
# - Implement a custom callback to log additional metrics and information during training
# 

# ----
# 

# ## Step-by-Step Instructions:
# 

# ### Exercise 1: Basic custom training loop: 
# 
# #### 1. Set Up the Environment:
# 
# - Import necessary libraries. 
# 
# - Load and preprocess the MNIST dataset. 
# 

# In[26]:


#get_ipython().system('pip install tensorflow numpy')


# In[27]:


import os
import warnings
import tensorflow as tf 
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.callbacks import Callback
import numpy as np

# Suppress all Python warnings
warnings.filterwarnings('ignore')

# Set TensorFlow log level to suppress warnings and info messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Step 1: Set Up the Environment
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data() 
x_train, x_test = x_train / 255.0, x_test / 255.0 
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)


# #### 2. Define the model: 
# 
# Create a simple neural network model with a Flatten layer followed by two Dense layers. 
# 

# In[28]:


# Step 2: Define the Model

model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10)
])


# #### 3. Define Loss Function and Optimizer: 
# 
# - Use Sparse Categorical Crossentropy for the loss function. 
# - Use the Adam optimizer. 
# 

# In[29]:


# Step 3: Define Loss Function and Optimizer

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) 
optimizer = tf.keras.optimizers.Adam()


# #### 4. Implement the Custom Training Loop: 
# 
# - Iterate over the dataset for a specified number of epochs. 
# - Compute the loss and apply gradients to update the model's weights. 
# 

# In[30]:


# Step 4: Implement the Custom Training Loop

epochs = 2
# train_dataset = train_dataset.repeat(epochs)
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
for epoch in range(epochs):
    print(f'Start of epoch {epoch + 1}')

    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            logits = model(x_batch_train, training=True)  # Forward pass
            loss_value = loss_fn(y_batch_train, logits)  # Compute loss

        # Compute gradients and update weights
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        # Logging the loss every 200 steps
        if step % 200 == 0:
            print(f'Epoch {epoch + 1} Step {step}: Loss = {loss_value.numpy()}')


# ### Exercise 2: Adding Accuracy Metric:
# 
# Enhance the custom training loop by adding an accuracy metric to monitor model performance. 
# 
# #### 1. Set Up the Environment: 
# 
# Follow the setup from Exercise 1. 
# 

# In[31]:


import tensorflow as tf 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Flatten 

# Step 1: Set Up the Environment
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize the pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0 

# Create a batched dataset for training
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)


# #### 2. Define the Model: 
# Use the same model as in Exercise 1. 
# 

# In[32]:


# Step 2: Define the Model

model = Sequential([ 
    Flatten(input_shape=(28, 28)),  # Flatten the input to a 1D vector
    Dense(128, activation='relu'),  # First hidden layer with 128 neurons and ReLU activation
    Dense(10)  # Output layer with 10 neurons for the 10 classes (digits 0-9)
])


# #### 3. Define the loss function, optimizer, and metric: 
# 
# - Use Sparse Categorical Crossentropy for the loss function and Adam optimizer. 
# 
# - Add Sparse Categorical Accuracy as a metric. 
# 

# In[33]:


# Step 3: Define Loss Function, Optimizer, and Metric

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)  # Loss function for multi-class classification
optimizer = tf.keras.optimizers.Adam()  # Adam optimizer for efficient training
accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy()  # Metric to track accuracy during training


# #### 4. Implement the custom training loop with accuracy: 
# 
# Track the accuracy during training and print it at regular intervals. 
# 

# In[34]:


# Step 4: Implement the Custom Training Loop with Accuracy

epochs = 5  # Number of epochs for training

for epoch in range(epochs):
    print(f'Start of epoch {epoch + 1}')
    
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            # Forward pass: Compute predictions
            logits = model(x_batch_train, training=True)
            # Compute loss
            loss_value = loss_fn(y_batch_train, logits)
        
        # Compute gradients
        grads = tape.gradient(loss_value, model.trainable_weights)
        # Apply gradients to update model weights
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        
        # Update the accuracy metric
        accuracy_metric.update_state(y_batch_train, logits)

        # Log the loss and accuracy every 200 steps
        if step % 200 == 0:
            print(f'Epoch {epoch + 1} Step {step}: Loss = {loss_value.numpy()} Accuracy = {accuracy_metric.result().numpy()}')
    
    # Reset the metric at the end of each epoch
    accuracy_metric.reset_state()


# ### Exercise 3: Custom Callback for Advanced Logging: 
# 
# Implement a custom callback to log additional metrics and information during training. 
# 
# #### 1. Set Up the Environment: 
# 
# Follow the setup from Exercise 1.
# 

# In[35]:


import tensorflow as tf 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Flatten 

# Step 1: Set Up the Environment
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize the pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0 

# Create a batched dataset for training
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)


# #### 2. Define the Model: 
# 
# Use the same model as in Exercise 1. 
# 

# In[36]:


# Step 2: Define the Model

model = Sequential([
    Flatten(input_shape=(28, 28)),  # Flatten the input to a 1D vector
    Dense(128, activation='relu'),  # First hidden layer with 128 neurons and ReLU activation
    Dense(10)  # Output layer with 10 neurons for the 10 classes (digits 0-9)
])


# #### 3. Define Loss Function, Optimizer, and Metric: 
# 
# - Use Sparse Categorical Crossentropy for the loss function and Adam optimizer. 
# 
# - Add Sparse Categorical Accuracy as a metric. 
# 

# In[37]:


# Step 3: Define Loss Function, Optimizer, and Metric

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)  # Loss function for multi-class classification
optimizer = tf.keras.optimizers.Adam()  # Adam optimizer for efficient training
accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy()  # Metric to track accuracy during training


# #### 4. Implement the custom training loop with custom callback: 
# 
# Create a custom callback to log additional metrics at the end of each epoch.
# 

# In[38]:


from tensorflow.keras.callbacks import Callback

# Step 4: Implement the Custom Callback 
class CustomCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print(f'End of epoch {epoch + 1}, loss: {logs.get("loss")}, accuracy: {logs.get("accuracy")}')


# In[39]:


# Step 5: Implement the Custom Training Loop with Custom Callback

epochs = 2
custom_callback = CustomCallback()  # Initialize the custom callback

for epoch in range(epochs):
    print(f'Start of epoch {epoch + 1}')
    
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            # Forward pass: Compute predictions
            logits = model(x_batch_train, training=True)
            # Compute loss
            loss_value = loss_fn(y_batch_train, logits)
        
        # Compute gradients
        grads = tape.gradient(loss_value, model.trainable_weights)
        # Apply gradients to update model weights
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        
        # Update the accuracy metric
        accuracy_metric.update_state(y_batch_train, logits)

        # Log the loss and accuracy every 200 steps
        if step % 200 == 0:
            print(f'Epoch {epoch + 1} Step {step}: Loss = {loss_value.numpy()} Accuracy = {accuracy_metric.result().numpy()}')
    
    # Call the custom callback at the end of each epoch
    custom_callback.on_epoch_end(epoch, logs={'loss': loss_value.numpy(), 'accuracy': accuracy_metric.result().numpy()})
    
    # Reset the metric at the end of each epoch
    accuracy_metric.reset_state()  # Use reset_state() instead of reset_states()


# ### Exercise 4: Add Hidden Layers 
# 
# Next, you will add a couple of hidden layers to your model. Hidden layers help the model learn complex patterns in the data. 
# 

# In[40]:


from tensorflow.keras.layers import Input, Dense

# Define the input layer
input_layer = Input(shape=(28, 28))  # Input layer with shape (28, 28)

# Define hidden layers
hidden_layer1 = Dense(64, activation='relu')(input_layer)  # First hidden layer with 64 neurons and ReLU activation
hidden_layer2 = Dense(64, activation='relu')(hidden_layer1)  # Second hidden layer with 64 neurons and ReLU activation


# In the above code: 
# 
# `Dense(64, activation='relu')` creates a dense (fully connected) layer with 64 units and ReLU activation function. 
# 
# Each hidden layer takes the output of the previous layer as its input.
# 

# ### Exercise 5: Define the output layer 
# 
# Finally, you will define the output layer. Suppose you are working on a binary classification problem, so the output layer will have one unit with a sigmoid activation function. 
# 

# In[41]:


output_layer = Dense(1, activation='sigmoid')(hidden_layer2)


# In the above code: 
# 
# `Dense(1, activation='sigmoid')` creates a dense layer with 1 unit and a sigmoid activation function, suitable for binary classification. 
# 

# ### Exercise 6: Create the Model 
# 
# Now, you will create the model by specifying the input and output layers. 
# 

# In[42]:


model = Model(inputs=input_layer, outputs=output_layer)


# In the above code: 
# 
# `Model(inputs=input_layer, outputs=output_layer)` creates a Keras model that connects the input layer to the output layer through the hidden layers. 
# 

# ### Exercise 7: Compile the Model 
# 
# Before training the model, you need to compile it. You will specify the loss function, optimizer, and evaluation metrics. 
# 

# In[43]:


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In the above code: 
# 
# `optimizer='adam'` specifies the Adam optimizer, a popular choice for training neural networks. 
# 
# `loss='binary_crossentropy'` specifies the loss function for binary classification problems. 
# 
# `metrics=['accuracy']` tells Keras to evaluate the model using accuracy during training. 
# 

# ### Exercise 8: Train the Model 
# 
# You can now train the model on some training data. For this example, let's assume `X_train` is our training input data and `y_train` is the corresponding labels. 
# 

# In[44]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import numpy as np

# Step 1: Redefine the Model for 20 features
model = Sequential([
    Input(shape=(20,)),  # Adjust input shape to (20,)
    Dense(128, activation='relu'),  # Hidden layer with 128 neurons and ReLU activation
    Dense(1, activation='sigmoid')  # Output layer for binary classification with sigmoid activation
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 2: Generate Example Data
X_train = np.random.rand(1000, 20)  # 1000 samples, 20 features each
y_train = np.random.randint(2, size=(1000, 1))  # 1000 binary labels (0 or 1)

# Step 3: Train the Model
model.fit(X_train, y_train, epochs=10, batch_size=32)


# In the above code: 
# 
# `X_train` and `y_train` are placeholders for your actual training data. 
# 
# `model.fit` trains the model for a specified number of epochs and batch size. 
# 

# ### Exercise 9: Evaluate the Model 
# 
# After training, you can evaluate the model on test data to see how well it performs. 
# 

# In[45]:


# Example test data (in practice, use real dataset)
X_test = np.random.rand(200, 20)  # 200 samples, 20 features each
y_test = np.random.randint(2, size=(200, 1))  # 200 binary labels (0 or 1)

# Evaluate the model on the test data
loss, accuracy = model.evaluate(X_test, y_test)

# Print test loss and accuracy
print(f'Test loss: {loss}')
print(f'Test accuracy: {accuracy}')


# In the above code: 
# 
# `model.evaluate` computes the loss and accuracy of the model on test data. 
# 
# `X_test` and `y_test` are placeholders for your actual test data. 
# 

# ## Practice Exercises 
# 
# ### Exercise 1: Basic Custom Training Loop 
# 
# #### Objective: Implement a basic custom training loop to train a simple neural network on the MNIST dataset. 
# 
# #### Instructions: 
# 
# - Set up the environment and load the dataset. 
# 
# - Define the model with a Flatten layer and two Dense layers. 
# 
# - Define the loss function and optimizer. 
# 
# - Implement a custom training loop to iterate over the dataset, compute the loss, and update the model's weights. 
# 

# In[46]:


# Write your code here
# Import necessary libraries
import tensorflow as tf 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Flatten 

# Step 1: Set Up the Environment
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data() 
x_train, x_test = x_train / 255.0, x_test / 255.0 
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32) 

# Step 2: Define the Model
model = Sequential([ 
    Flatten(input_shape=(28, 28)), 
    Dense(128, activation='relu'), 
    Dense(10) 
]) 

# Step 3: Define Loss Function and Optimizer
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) 
optimizer = tf.keras.optimizers.Adam() 

# Step 4: Implement the Custom Training Loop
for epoch in range(5): 
    for x_batch, y_batch in train_dataset: 
        with tf.GradientTape() as tape: 
            logits = model(x_batch, training=True) 
            loss = loss_fn(y_batch, logits) 
        grads = tape.gradient(loss, model.trainable_weights) 
        optimizer.apply_gradients(zip(grads, model.trainable_weights)) 
    print(f'Epoch {epoch + 1}: Loss = {loss.numpy()}')


# <details>
# <summary>Click here for solution</summary> </br>
# 
# ```python
# # Import necessary libraries
# import tensorflow as tf 
# from tensorflow.keras.models import Sequential 
# from tensorflow.keras.layers import Dense, Flatten 
# 
# # Step 1: Set Up the Environment
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data() 
# x_train, x_test = x_train / 255.0, x_test / 255.0 
# train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32) 
# 
# # Step 2: Define the Model
# model = Sequential([ 
#     Flatten(input_shape=(28, 28)), 
#     Dense(128, activation='relu'), 
#     Dense(10) 
# ]) 
# 
# # Step 3: Define Loss Function and Optimizer
# loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) 
# optimizer = tf.keras.optimizers.Adam() 
# 
# # Step 4: Implement the Custom Training Loop
# for epoch in range(5): 
#     for x_batch, y_batch in train_dataset: 
#         with tf.GradientTape() as tape: 
#             logits = model(x_batch, training=True) 
#             loss = loss_fn(y_batch, logits) 
#         grads = tape.gradient(loss, model.trainable_weights) 
#         optimizer.apply_gradients(zip(grads, model.trainable_weights)) 
#     print(f'Epoch {epoch + 1}: Loss = {loss.numpy()}')
# 

# ### Exercise 2: Adding Accuracy Metric 
# 
# #### Objective: Enhance the custom training loop by adding an accuracy metric to monitor model performance. 
# 
# #### Instructions: 
# 
# 1. Set up the environment and define the model, loss function, and optimizer. 
# 
# 2. Add Sparse Categorical Accuracy as a metric. 
# 
# 3. Implement the custom training loop with accuracy tracking.
# 

# In[47]:


# Write your code here
# Import necessary libraries
import tensorflow as tf 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Flatten 

# Step 1: Set Up the Environment
(x_train, y_train), _ = tf.keras.datasets.mnist.load_data() 
x_train = x_train / 255.0 
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32) 

# Step 2: Define the Model
model = Sequential([ 
    Flatten(input_shape=(28, 28)), 
    Dense(128, activation='relu'), 
    Dense(10) 
]) 

# Step 3: Define Loss Function, Optimizer, and Metric
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) 
optimizer = tf.keras.optimizers.Adam() 
accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy() 

# Step 4: Implement the Custom Training Loop with Accuracy Tracking
epochs = 5 
for epoch in range(epochs): 
    for x_batch, y_batch in train_dataset: 
        with tf.GradientTape() as tape: 
            logits = model(x_batch, training=True) 
            loss = loss_fn(y_batch, logits) 
        grads = tape.gradient(loss, model.trainable_weights) 
        optimizer.apply_gradients(zip(grads, model.trainable_weights)) 
        accuracy_metric.update_state(y_batch, logits) 
    print(f'Epoch {epoch + 1}: Loss = {loss.numpy()} Accuracy = {accuracy_metric.result().numpy()}') 
    accuracy_metric.reset_state()


# <details>
# <summary>Click here for solution</summary><br>
# 
# ```python
# # Import necessary libraries
# import tensorflow as tf 
# from tensorflow.keras.models import Sequential 
# from tensorflow.keras.layers import Dense, Flatten 
# 
# # Step 1: Set Up the Environment
# (x_train, y_train), _ = tf.keras.datasets.mnist.load_data() 
# x_train = x_train / 255.0 
# train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32) 
# 
# # Step 2: Define the Model
# model = Sequential([ 
#     Flatten(input_shape=(28, 28)), 
#     Dense(128, activation='relu'), 
#     Dense(10) 
# ]) 
# 
# # Step 3: Define Loss Function, Optimizer, and Metric
# loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) 
# optimizer = tf.keras.optimizers.Adam() 
# accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy() 
# 
# # Step 4: Implement the Custom Training Loop with Accuracy Tracking
# epochs = 5 
# for epoch in range(epochs): 
#     for x_batch, y_batch in train_dataset: 
#         with tf.GradientTape() as tape: 
#             logits = model(x_batch, training=True) 
#             loss = loss_fn(y_batch, logits) 
#         grads = tape.gradient(loss, model.trainable_weights) 
#         optimizer.apply_gradients(zip(grads, model.trainable_weights)) 
#         accuracy_metric.update_state(y_batch, logits) 
#     print(f'Epoch {epoch + 1}: Loss = {loss.numpy()} Accuracy = {accuracy_metric.result().numpy()}') 
#     accuracy_metric.reset_state() 
# 

# ### Exercise 3: Custom Callback for Advanced Logging 
# 
# #### Objective: Implement a custom callback to log additional metrics and information during training. 
# 
# #### Instructions: 
# 
# 1. Set up the environment and define the model, loss function, optimizer, and metric. 
# 
# 2. Create a custom callback to log additional metrics at the end of each epoch. 
# 
# 3. Implement the custom training loop with the custom callback. 
# 

# In[48]:


# Write your code here
# Import necessary libraries
import tensorflow as tf 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Flatten 
from tensorflow.keras.callbacks import Callback 

# Step 1: Set Up the Environment
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data() 
x_train = x_train / 255.0 
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32) 

# Step 2: Define the Model
model = Sequential([ 
    tf.keras.Input(shape=(28, 28)),  # Updated Input layer syntax
    Flatten(), 
    Dense(128, activation='relu'), 
    Dense(10) 
]) 

# Step 3: Define Loss Function, Optimizer, and Metric
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) 
optimizer = tf.keras.optimizers.Adam() 
accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy() 

# Step 4: Implement the Custom Callback
class CustomCallback(Callback): 
    def on_epoch_end(self, epoch, logs=None): 
        print(f'End of epoch {epoch + 1}, loss: {logs.get("loss")}, accuracy: {logs.get("accuracy")}') 

# Step 5: Implement the Custom Training Loop with Custom Callback
custom_callback = CustomCallback() 

for epoch in range(5): 
    for x_batch, y_batch in train_dataset: 
        with tf.GradientTape() as tape: 
            logits = model(x_batch, training=True) 
            loss = loss_fn(y_batch, logits) 
        grads = tape.gradient(loss, model.trainable_weights) 
        optimizer.apply_gradients(zip(grads, model.trainable_weights)) 
        accuracy_metric.update_state(y_batch, logits) 
    custom_callback.on_epoch_end(epoch, logs={'loss': loss.numpy(), 'accuracy': accuracy_metric.result().numpy()}) 
    accuracy_metric.reset_state()  # Updated method


# <details>
# <summary>Click here for solution</summary> </br>
# 
# ```python
# # Import necessary libraries
# import tensorflow as tf 
# from tensorflow.keras.models import Sequential 
# from tensorflow.keras.layers import Dense, Flatten 
# from tensorflow.keras.callbacks import Callback 
# 
# # Step 1: Set Up the Environment
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data() 
# x_train = x_train / 255.0 
# train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32) 
# 
# # Step 2: Define the Model
# model = Sequential([ 
#     tf.keras.Input(shape=(28, 28)),  # Updated Input layer syntax
#     Flatten(), 
#     Dense(128, activation='relu'), 
#     Dense(10) 
# ]) 
# 
# # Step 3: Define Loss Function, Optimizer, and Metric
# loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) 
# optimizer = tf.keras.optimizers.Adam() 
# accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy() 
# 
# # Step 4: Implement the Custom Callback
# class CustomCallback(Callback): 
#     def on_epoch_end(self, epoch, logs=None): 
#         print(f'End of epoch {epoch + 1}, loss: {logs.get("loss")}, accuracy: {logs.get("accuracy")}') 
# 
# # Step 5: Implement the Custom Training Loop with Custom Callback
# custom_callback = CustomCallback() 
# 
# for epoch in range(5): 
#     for x_batch, y_batch in train_dataset: 
#         with tf.GradientTape() as tape: 
#             logits = model(x_batch, training=True) 
#             loss = loss_fn(y_batch, logits) 
#         grads = tape.gradient(loss, model.trainable_weights) 
#         optimizer.apply_gradients(zip(grads, model.trainable_weights)) 
#         accuracy_metric.update_state(y_batch, logits) 
#     custom_callback.on_epoch_end(epoch, logs={'loss': loss.numpy(), 'accuracy': accuracy_metric.result().numpy()}) 
#     accuracy_metric.reset_state()  # Updated method
# 
# 

# ### Exercise 4: Lab - Hyperparameter Tuning 
# 
# #### Enhancement: Add functionality to save the results of each hyperparameter tuning iteration as JSON files in a specified directory. 
# 
# #### Additional Instructions:
# 
# Modify the tuning loop to save each iteration's results as JSON files.
# 
# Specify the directory where these JSON files will be stored for easier retrieval and analysis of tuning results.
# 

# In[50]:


# Write your code here
#get_ipython().system('pip install keras-tuner')
#get_ipython().system('pip install scikit-learn')

import json
import os
import keras_tuner as kt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# Step 1: Load your dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

# Step 2: Define the model-building function
def build_model(hp):
    model = Sequential()
    # Tune the number of units in the first Dense layer
    model.add(Dense(units=hp.Int('units', min_value=32, max_value=512, step=32),
                    activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification example
    model.compile(optimizer=Adam(hp.Float('learning_rate', 1e-4, 1e-2, sampling='LOG')),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# Step 3: Initialize a Keras Tuner RandomSearch tuner
tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=10,  # Set the number of trials
    executions_per_trial=1,  # Set how many executions per trial
    directory='tuner_results',  # Directory for saving logs
    project_name='hyperparam_tuning'
)

# Step 4: Run the tuner search (make sure the data is correct)
tuner.search(X_train, y_train, validation_data=(X_val, y_val), epochs=5)

# Step 5: Save the tuning results as JSON files
try:
    for i in range(10):
        # Fetch the best hyperparameters from the tuner
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        
        # Results dictionary to save hyperparameters and score
        results = {
            "trial": i + 1,
            "hyperparameters": best_hps.values,  # Hyperparameters tuned in this trial
            "score": None  # Add any score or metrics if available
        }

        # Save the results as JSON
        with open(os.path.join('tuner_results', f"trial_{i + 1}.json"), "w") as f:
            json.dump(results, f)

except IndexError:
    print("Tuning process has not completed or no results available.")


# <details>
# <summary>Click here for solution</summary> </br>
# 
# ```python
# !pip install keras-tuner
# !pip install scikit-learn
# 
# import json
# import os
# import keras_tuner as kt
# from tensorflow.keras import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.optimizers import Adam
# from sklearn.model_selection import train_test_split
# from sklearn.datasets import make_classification
# 
# # Step 1: Load your dataset
# X, y = make_classification(n_samples=1000, n_features=20, n_classes=2)
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
# 
# # Step 2: Define the model-building function
# def build_model(hp):
#     model = Sequential()
#     # Tune the number of units in the first Dense layer
#     model.add(Dense(units=hp.Int('units', min_value=32, max_value=512, step=32),
#                     activation='relu'))
#     model.add(Dense(1, activation='sigmoid'))  # Binary classification example
#     model.compile(optimizer=Adam(hp.Float('learning_rate', 1e-4, 1e-2, sampling='LOG')),
#                   loss='binary_crossentropy',
#                   metrics=['accuracy'])
#     return model
# 
# # Step 3: Initialize a Keras Tuner RandomSearch tuner
# tuner = kt.RandomSearch(
#     build_model,
#     objective='val_accuracy',
#     max_trials=10,  # Set the number of trials
#     executions_per_trial=1,  # Set how many executions per trial
#     directory='tuner_results',  # Directory for saving logs
#     project_name='hyperparam_tuning'
# )
# 
# # Step 4: Run the tuner search (make sure the data is correct)
# tuner.search(X_train, y_train, validation_data=(X_val, y_val), epochs=5)
# 
# # Step 5: Save the tuning results as JSON files
# try:
#     for i in range(10):
#         # Fetch the best hyperparameters from the tuner
#         best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
#         
#         # Results dictionary to save hyperparameters and score
#         results = {
#             "trial": i + 1,
#             "hyperparameters": best_hps.values,  # Hyperparameters tuned in this trial
#             "score": None  # Add any score or metrics if available
#         }
# 
#         # Save the results as JSON
#         with open(os.path.join('tuning_results', f"trial_{i + 1}.json"), "w") as f:
#             json.dump(results, f)
# 
# except IndexError:
#     print("Tuning process has not completed or no results available.")
#  ```   
# 
# </details>
# 

# ### Exercise 5: Explanation of Hyperparameter Tuning
# 
# **Addition to Explanation:** Add a note explaining the purpose of num_trials in the hyperparameter tuning context:
# 

# In[51]:


# Write your code here
Explanation: "num_trials specifies the number of top hyperparameter sets to return. Setting num_trials=1 means that it will return only the best set of hyperparameters found during the tuning process."


# <details>
# <summary>Click here for solution</summary> </br>
# 
# ```python
# Explanation: "num_trials specifies the number of top hyperparameter sets to return. Setting num_trials=1 means that it will return only the best set of hyperparameters found during the tuning process."
#  ```   
# 
# </details>
# 

# ### Conclusion: 
# 
# Congratulations on completing this lab! You have now successfully created, trained, and evaluated a simple neural network model using the Keras Functional API. This foundational knowledge will allow you to build more complex models and explore advanced functionalities in Keras. 
# 

# Copyright © IBM Corporation. All rights reserved.
# 

# In[ ]:




