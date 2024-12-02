# jupyter nbconvert --to script "Lab_Practical Application of Transpose Convolution.ipynb"
# #!/usr/bin/env python
# coding: utf-8

# <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/BQm_EV6i0_j80CQZ8vcLvw/SN-web-lightmode.png" width=300 height=300 />
# 

# ## Lab: Practical Application of Transpose Convolution  
# 
# **Estimated time needed: 30 minutes**
# 
# This lab will guide you through creating, training, and evaluating models that use transpose convolution layers for tasks such as image reconstruction. 
# 
# ##### Learning objectives: 
# 
# By the end of this lab, you will: 
# 
# - Apply transpose convolution in practical scenarios using Keras.  
# 
# - Create, compile, train, and evaluate the model 
# 
# - Visualize the results  
# 
# ##### Prerequisites: 
# 
# - Basic understanding of Python programming 
# 
# - Familiarity with convolutional neural network (CNN) concepts 
# 
# - Keras and TensorFlow installed 
# 
# 
# 

# #### Steps: 
# **Step 1: Import Necessary Libraries**
# 
# Before you start, make sure to import the required libraries: TensorFlow and Keras. Keras is included within TensorFlow as `tensorflow.keras`. 
# 

# In[1]:


import warnings
warnings.simplefilter('ignore')

#get_ipython().system('pip install tensorflow==2.16.2')

#get_ipython().system('pip install matplotlib')
import tensorflow as tf 

from tensorflow.keras.models import Model 

from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, UpSampling2D 

import numpy as np 

import matplotlib.pyplot as plt 


# **Explanation:**
# 
#  `tensorflow` is the main library for machine learning in Python. 
# 
#   `Model, Input, Conv2D, and Conv2DTranspose` are used to create and manipulate the layers in your model. 
# 
#   `numpy` and `matplotlib` are used for data manipulation and visualization. 
# 
#  
# **Step 2: Define the Input Layer**
# 
# You need to define the input shape for your model. For simplicity, let's assume you are working with an input image of size 28x28 with 1 channel (grayscale). 
# 
#  
# 

# In[2]:


input_layer = Input(shape=(28, 28, 1))


# **Explanation:**
# 
# `Input(shape=(28, 28, 1))` creates an input layer that expects 28x28 grayscale images.
# 

# **Step 3: Add convolutional and transpose convolutional layers**
# 
# You will add a convolutional layer to extract features, followed by a transpose convolution layer to reconstruct the image. 
# 

# In[3]:


conv_layer = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer) 

transpose_conv_layer = Conv2DTranspose(filters=1, kernel_size=(3, 3), activation='sigmoid', padding='same')(conv_layer) 


# **Explanation:**
# 
# `Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')` creates a convolution layer with 32 filters. 
# 
# `Conv2DTranspose(filters=1, kernel_size=(3, 3), activation='sigmoid', padding='same')` creates a transpose convolution layer for reconstructing the image. 
# 
# **Step 4: Create the Model**
# 
# Now, you will create the model by specifying the input and output layers.
# 

# In[4]:


model = Model(inputs=input_layer, outputs=transpose_conv_layer)


# **Explanation:**
# 
# `Model(inputs=input_layer, outputs=transpose_conv_layer)` creates a Keras model that connects the input layer to the output layer through the convolutional and transpose convolutional layers.  
# 
# **Step 5: Compile the Model**
# 
# Before training the model, you need to compile it. You will specify the loss function, optimizer, and evaluation metrics. 
# 

# In[5]:


model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])


# **Explanation:**
# 
# - optimizer=`adam` specifies the Adam optimizer. 
# 
# - loss=`binary_crossentropy` specifies the binary cross-entropy loss function, suitable for binary pixel classification. 
# 
# - metrics=[`accuracy`] directs Keras to evaluate the model using accuracy during training. 
# 

# **Step 6: Train the Model**
# 
# You can now train the model on some training data. For this example, let's assume X_train is our training input data.
# 

# In[6]:


# Generate synthetic training data 

X_train = np.random.rand(1000, 28, 28, 1) 

y_train = X_train # For reconstruction, the target is the input 

history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2) 


# **Explanation:**
# 
# - X_train is a placeholder for your actual training data. 
# 
# - y_train is the same as X_train for image reconstruction. 
# 
# - model.fit trains the model for a specified number of epochs and batch size.
# 
#   
# **Step 7: Evaluate the Model**
# 
# After training, you can evaluate the model on test data to see how well it performs. 
# 

# In[7]:


# Generate synthetic test data 

X_test = np.random.rand(200, 28, 28, 1) 

y_test = X_test 


loss, accuracy = model.evaluate(X_test, y_test) 

print(f'Test loss: {loss}') 

print(f'Test accuracy: {accuracy}') 


# **Explanation:**
# 
# `X_test` and `y_test` are placeholders for your actual test data. 
# 
# `model.evaluate` computes the loss and accuracy of the model on test data. . 
# 
# **Step 8: Visualize the Results**
# 
# To better understand the model's performance, let's visualize some reconstructed images.  
# 

# In[8]:


# Predict on test data 
y_pred = model.predict(X_test) 

# Plot some sample images 

n = 10 # Number of samples to display 

plt.figure(figsize=(20, 4))

for i in range(n): 

    # Display original 
    ax = plt.subplot(2, n, i + 1) 
    plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
    plt.title("Original") 
    plt.axis('off') 
    # Display reconstruction 
    ax = plt.subplot(2, n, i + 1 + n) 
    plt.imshow(y_pred[i].reshape(28, 28), cmap='gray')
    plt.title("Reconstructed")
    plt.axis('off')

plt.show() 


# **Explanation:**
# 
# `model.predict` generates predictions for the test data. 
# 
# `plt.imshow` visualizes the original and reconstructed images side by side for comparison. 
# 

# ### Practice exercises
# 

# #### Exercise 1: Experiment with Different Kernel Sizes 
# 
# **Objective:** Understand the impact of different kernel sizes on the model's performance. 
# 
# **Instructions:**
# 1. Modify the kernel size of the `Conv2D` and `Conv2DTranspose` layers. 
# 2. Recompile, train, and evaluate the model. 
# 3. Observe and record the differences in performance.
# 

# In[9]:


# Write your code here
from tensorflow.keras.layers import Dropout, Conv2D, Conv2DTranspose, Input
from tensorflow.keras.models import Model

# Define the input layer
input_layer = Input(shape=(28, 28, 1))

# Add convolutional and transpose convolutional layers with different kernel sizes
conv_layer = Conv2D(filters=32, kernel_size=(5, 5), activation='relu', padding='same')(input_layer)
transpose_conv_layer = Conv2DTranspose(filters=1, kernel_size=(5, 5), activation='sigmoid', padding='same')(conv_layer)

# Create the model
model = Model(inputs=input_layer, outputs=transpose_conv_layer)

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test loss: {loss}')
print(f'Test accuracy: {accuracy}')
# why random data rather than real image !


# <details>
# <summary>Click here for solution</summary> </br>
# 
# ```python
# from tensorflow.keras.layers import Dropout, Conv2D, Conv2DTranspose, Input
# from tensorflow.keras.models import Model
# 
# # Define the input layer
# input_layer = Input(shape=(28, 28, 1))
# 
# # Add convolutional and transpose convolutional layers with different kernel sizes
# conv_layer = Conv2D(filters=32, kernel_size=(5, 5), activation='relu', padding='same')(input_layer)
# transpose_conv_layer = Conv2DTranspose(filters=1, kernel_size=(5, 5), activation='sigmoid', padding='same')(conv_layer)
# 
# # Create the model
# model = Model(inputs=input_layer, outputs=transpose_conv_layer)
# 
# # Compile the model
# model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
# 
# # Train the model
# history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
# 
# # Evaluate the model
# loss, accuracy = model.evaluate(X_test, y_test)
# print(f'Test loss: {loss}')
# print(f'Test accuracy: {accuracy}')
# 
# 

# #### Exercise 2: Add Dropout Layers
# 
# **Objective:** Add dropout layers to prevent overfitting.
# 
# **Instructions:**
# 1. Add dropout layers after the convolutional layer. 
# 2. Set the dropout rate to 0.5.
# 3. Recompile, train, and evaluate the model.
# 

# In[10]:


# Write your code here
from tensorflow.keras.layers import Dropout, Conv2D, Conv2DTranspose, Input
from tensorflow.keras.models import Model

# Define the input layer
input_layer = Input(shape=(28, 28, 1))

# Add convolutional, dropout, and transpose convolutional layers
conv_layer = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
dropout_layer = Dropout(0.5)(conv_layer)
transpose_conv_layer = Conv2DTranspose(filters=1, kernel_size=(3, 3), activation='sigmoid', padding='same')(dropout_layer)

# Create the model
model = Model(inputs=input_layer, outputs=transpose_conv_layer)

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test loss: {loss}')
print(f'Test accuracy: {accuracy}')
# why random data rather than real image !


# <details>
# <summary>Click here for solution</summary> </br>
# 
# ```python
# from tensorflow.keras.layers import Dropout, Conv2D, Conv2DTranspose, Input
# from tensorflow.keras.models import Model
# 
# # Define the input layer
# input_layer = Input(shape=(28, 28, 1))
# 
# # Add convolutional, dropout, and transpose convolutional layers
# conv_layer = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer)
# dropout_layer = Dropout(0.5)(conv_layer)
# transpose_conv_layer = Conv2DTranspose(filters=1, kernel_size=(3, 3), activation='sigmoid', padding='same')(dropout_layer)
# 
# # Create the model
# model = Model(inputs=input_layer, outputs=transpose_conv_layer)
# 
# # Compile the model
# model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
# 
# # Train the model
# history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
# 
# # Evaluate the model
# loss, accuracy = model.evaluate(X_test, y_test)
# print(f'Test loss: {loss}')
# print(f'Test accuracy: {accuracy}')
# 
# 

# #### Exercise 3: Use Different Activation Functions 
# 
# **Objective:** Experiment with different activation functions and observe their impact on model performance. 
# 
# **Instructions:**
# 1. Change the activation function of the convolutional and transpose convolutional layers to `tanh`. 
# 2. Recompile, train, and evaluate the model.
# 

# In[ ]:


# Write your code here


# <details>
# <summary>Click here for solution</summary> </br>
# 
# ```python
# 
# from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Input
# from tensorflow.keras.models import Model
# 
# # Define the input layer
# input_layer = Input(shape=(28, 28, 1))
# 
# # Add convolutional and transpose convolutional layers with different activation functions
# conv_layer = Conv2D(filters=32, kernel_size=(3, 3), activation='tanh', padding='same')(input_layer)
# transpose_conv_layer = Conv2DTranspose(filters=1, kernel_size=(3, 3), activation='tanh', padding='same')(conv_layer)
# 
# # Create the model
# model = Model(inputs=input_layer, outputs=transpose_conv_layer)
# 
# # Compile the model
# model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
# 
# # Train the model
# history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
# 
# # Evaluate the model
# loss, accuracy = model.evaluate(X_test, y_test)
# print(f'Test loss: {loss}')
# print(f'Test accuracy: {accuracy}')
# 

# ### Conclusion:
# 
# By completing this lab, you have successfully created, trained, and evaluated a simple neural network model using transpose convolution for image reconstruction. This exercise provided hands-on experience with Keras and practical applications of transpose convolution layers. Continue experimenting with different architectures and datasets to deepen your understanding and skills in deep learning with Keras. 
# 

# Copyright © IBM Corporation. All rights reserved.
# 
