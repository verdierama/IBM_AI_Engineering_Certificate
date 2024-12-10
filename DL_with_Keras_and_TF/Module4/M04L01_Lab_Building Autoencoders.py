# jupyter nbconvert --to script "M04L01_Lab_Building Autoencoders.ipynb"
# #!/usr/bin/env python
# coding: utf-8

# <p style="text-align:center">
#     <a href="https://skills.network" target="_blank">
#     <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/assets/logos/SN_web_lightmode.png" width="200" alt="Skills Network Logo"  />
#     </a>
# </p>
# 

# # **Lab: Building Autoencoders**
# 

# Estimated time needed: **30** minutes
# 

# In this lab, you will learn how to build autoencoders using Keras.  
# 

# ## Learning Objectives
# 
# By the end of this lab, you will: 
# 
# - Load and preprocess the MNIST dataset for training an autoencoder. 
# 
# - Construct a simple autoencoder model using the Keras functional API. 
# 
# - Train the autoencoder on the MNIST dataset. 
# 
# - Evaluate the performance of the trained autoencoder. 
# 
# - Fine-tune the autoencoder to improve its performance. 
# 
# - Use the autoencoder to denoise images. 
# 

# ----
# 

# ### Step-by-Step Instructions: 
# 
# #### Step 1: Data Preprocessing 
# 
# This exercise prepares the MNIST dataset for training by normalizing the pixel values and flattening the images. Normalization helps in faster convergence during training, and flattening is required because the input layer of our autoencoder expects a one-dimensional vector. 
# 

# In[1]:


#get_ipython().system('pip install tensorflow==2.16.2')


# In[2]:


import numpy as np 
from tensorflow.keras.datasets import mnist 

# Load the dataset 
(x_train, _), (x_test, _) = mnist.load_data() 

# Normalize the pixel values 
x_train = x_train.astype('float32') / 255. 
x_test = x_test.astype('float32') / 255. 

# Flatten the images 
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:]))) 
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:]))) 


# In the above code: 
# - Use Keras to load the MNIST dataset. 
# - Normalize the image pixel values to the range [0, 1]. 
# - Flatten the 28x28 images to a 784-dimensional vector to reshape the data. 
# 

# #### Step 2: Building the Autoencoder Model 
# 
# This exercise involves building an autoencoder with an encoder that compresses the input to 32 dimensions and a decoder that reconstructs the input from these 32 dimensions. The model is compiled with the Adam optimizer and binary crossentropy loss. 
# 

# In[3]:


from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Input, Dense 

# Encoder 
input_layer = Input(shape=(784,)) 
encoded = Dense(64, activation='relu')(input_layer) 

# Bottleneck 
bottleneck = Dense(32, activation='relu')(encoded) 

# Decoder 
decoded = Dense(64, activation='relu')(bottleneck) 
output_layer = Dense(784, activation='sigmoid')(decoded) 

# Autoencoder model 
autoencoder = Model(input_layer, output_layer) 

# Compile the model 
autoencoder.compile(optimizer='adam', loss='binary_crossentropy') 

# Summary of the model 
autoencoder.summary() 


# In the above code: 
# 
# **1. Define the Encoder:**
# - Create an input layer with 784 neurons. 
# - Add a Dense layer with 64 neurons and ReLU activation. 
# 
# **2. Define the Bottleneck:**
# - Add a Dense layer with 32 neurons and ReLU activation. 
# 
# **3. Define the Decoder:**
# - Add a Dense layer with 64 neurons and ReLU activation. 
# - Add an output layer with 784 neurons and sigmoid activation. 
# 
# **4. Compile the Model:**
# - Use the Adam optimizer and binary crossentropy loss.  
# 

# #### Step 3: Training the Autoencoder 
# 
# In this exercise, the autoencoder is trained to reconstruct the MNIST images. The training data is both the input and the target, as the autoencoder learns to map the input to itself. 
# 

# In[4]:


autoencoder.fit(
    x_train, x_train,  
    epochs=25,  
    batch_size=256,  
    shuffle=True,  
    validation_data=(x_test, x_test)
)


# In the above code: 
# - Use the `fit` method to train the model on the training data. 
# - Set the number of epochs to 25 and the batch size to 256.. 
# - Use the test data for validation. 
# 

# #### Step 4: Evaluating the Autoencoder 
# 
# This exercise evaluates the autoencoder by reconstructing the test images and comparing them to the original images. Visualization helps in understanding how well the autoencoder has learned to reconstruct the input data. 
# 

# In[5]:


#get_ipython().system('pip install matplotlib==3.9.2')


# In[6]:


import matplotlib.pyplot as plt 

# Predict the test data 
reconstructed = autoencoder.predict(x_test) 

# Visualize the results 
n = 10  # Number of digits to display 
plt.figure(figsize=(20, 4)) 

for i in range(n): 
    # Display original 
    ax = plt.subplot(2, n, i + 1) 
    plt.imshow(x_test[i].reshape(28, 28)) 
    plt.gray() 
    ax.get_xaxis().set_visible(False) 
    ax.get_yaxis().set_visible(False) 

    # Display reconstruction 
    ax = plt.subplot(2, n, i + 1 + n) 
    plt.imshow(reconstructed[i].reshape(28, 28)) 
    plt.gray() 
    ax.get_xaxis().set_visible(False) 
    ax.get_yaxis().set_visible(False) 

plt.show()


# In the above code: 
# 
# **1. Reconstruct Images:**
# - Use the autoencoder to predict the test data. 
# - Compare the original test images with the reconstructed images. 
# 
# **2. Visualize the Results:**
# - Plot a few examples of original and reconstructed images side by side. 
# 

# #### Step 5: Fine-Tuning the Autoencoder 
# 
# Fine-tuning the autoencoder by unfreezing some layers can help in improving its performance. In this exercise, you unfreeze the last four layers and train the model again for a few more epochs.
# 

# In[7]:


# Unfreeze the top layers of the encoder
for layer in autoencoder.layers[-4:]: 
    layer.trainable = True 

# Compile the model again
autoencoder.compile(optimizer='adam', loss='binary_crossentropy') 

# Train the model again
autoencoder.fit(x_train, x_train,  
                epochs=10,  
                batch_size=256,  
                shuffle=True,  
                validation_data=(x_test, x_test))


# In the above code: 
# 
# **1. Unfreeze the Encoder Layers:**
# - Unfreeze the last four layers of the encoder. 
# 
# **2. Compile and Train the Model:**
# - Recompile the model. 
# - Train the model again for 10 epochs with the same training and validation data.
# 

# #### Step 6: Denoising Images with Autoencoder 
# 
# In this exercise, you add random noise to the dataset and train the autoencoder to denoise the images. The autoencoder learns to reconstruct the original images from the noisy input, which can be visualized by comparing the noisy, denoised, and original images. 
# 

# In[9]:


import numpy as np
import matplotlib.pyplot as plt

# Add noise to the data
noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

# Train the autoencoder with noisy data
autoencoder.fit(
    x_train_noisy, x_train,
    epochs=20,
    batch_size=512,
    shuffle=True,
    validation_data=(x_test_noisy, x_test)
)

# Denoise the test images
reconstructed_noisy = autoencoder.predict(x_test_noisy)

# Visualize the results
n = 10  # Number of digits to display
plt.figure(figsize=(20, 6))
for i in range(n):
    # Display noisy images
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(x_test_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    
    # Display denoised images
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(reconstructed_noisy[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display original images
    ax = plt.subplot(3, n, i + 1 + 2 * n)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()


# In the above code: 
# 
# **1. Add noise to the data:**
# - Add random noise to the training and test data. 
# - Train the Autoencoder with noisy data: 
# -Train the autoencoder using the noisy images as input and the original images as target. 
# 
# **2. Evaluate the denoising performance:**
# - Use the autoencoder to denoise the test images. 
# - Compare the noisy, denoised, and original images. 
# 

# ## Practice Exercises: 
# 
# ### Exercise 1: Exploring Different Bottleneck Sizes 
# 
# #### Objective: 
# 
# To understand the impact of different bottleneck sizes on the performance of the autoencoder. 
# 
# #### Instructions: 
# 
# **1. Define new models with different bottleneck sizes:**
# - Create three new autoencoder models, each with a different bottleneck size (e.g., 16, 32, and 64 neurons). 
# - Use the same encoder and decoder architecture as in the main lab but change the number of neurons in the bottleneck layer. 
# 
# **2. Train the models:**
# - Train each model on the MNIST dataset for 50 epochs with a batch size of 256. 
# - Use the same preprocessing steps as in the main lab. 
# 
# **3. Evaluate and Compare the Models:**
# - Evaluate the performance of each model on the test data. 
# - Compare the reconstruction loss of the models to understand how the bottleneck size affects the autoencoder's ability to reconstruct the input data. 
# 

# In[13]:


# Write your code here
# Encoder 
input_layer = Input(shape=(784,)) 
encoded = Dense(64, activation='relu')(input_layer) 

# Bottleneck 
bottleneck16 = Dense(16, activation='relu')(encoded)
bottleneck32 = Dense(32, activation='relu')(encoded) 
bottleneck64 = Dense(64, activation='relu')(encoded) 

# Decoder 
decoded16 = Dense(64, activation='relu')(bottleneck16) 
decoded32 = Dense(64, activation='relu')(bottleneck32) 
decoded64 = Dense(64, activation='relu')(bottleneck64) 
output_layer16 = Dense(784, activation='sigmoid')(decoded16)
output_layer32 = Dense(784, activation='sigmoid')(decoded32) 
output_layer64 = Dense(784, activation='sigmoid')(decoded64) 

# Autoencoder model 
autoencoder16 = Model(input_layer, output_layer16)
autoencoder32 = Model(input_layer, output_layer32) 
autoencoder64 = Model(input_layer, output_layer64) 

# Compile the model 
autoencoder16.compile(optimizer='adam', loss='binary_crossentropy') 
autoencoder32.compile(optimizer='adam', loss='binary_crossentropy') 
autoencoder64.compile(optimizer='adam', loss='binary_crossentropy') 

# Summary of the model 
autoencoder16.summary() 
autoencoder32.summary() 
autoencoder64.summary() 

autoencoder16.fit(
    x_train, x_train,  
    epochs=50,  
    batch_size=256,  
    shuffle=True,  
    validation_data=(x_test, x_test)
)

autoencoder32.fit(
    x_train, x_train,  
    epochs=50,  
    batch_size=256,  
    shuffle=True,  
    validation_data=(x_test, x_test)
)

autoencoder64.fit(
    x_train, x_train,  
    epochs=50,  
    batch_size=256,  
    shuffle=True,  
    validation_data=(x_test, x_test)
)


loss16 = autoencoder16.evaluate(x_test, x_test)
print(f'Bottleneck size 16 - Test loss: {loss16}')
loss32 = autoencoder32.evaluate(x_test, x_test)
print(f'Bottleneck size 32 - Test loss: {loss32}')
loss64 = autoencoder64.evaluate(x_test, x_test)
print(f'Bottleneck size 64 - Test loss: {loss64}')


# <details>
#     <summary>Click here for Solution</summary>
# 
# ```python
# # Define and train three different autoencoders with varying bottleneck sizes
# bottleneck_sizes = [16, 32, 64]
# autoencoders = []
# 
# for size in bottleneck_sizes:
#     # Encoder
#     input_layer = Input(shape=(784,))
#     encoded = Dense(64, activation='relu')(input_layer)
#     bottleneck = Dense(size, activation='relu')(encoded)
# 
#     # Decoder
#     decoded = Dense(64, activation='relu')(bottleneck)
#     output_layer = Dense(784, activation='sigmoid')(decoded)
# 
#     # Autoencoder model
#     autoencoder = Model(input_layer, output_layer)
#     autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
#     autoencoder.fit(
#         x_train,
#         x_train,
#         epochs=20,
#         batch_size=256,
#         shuffle=True,
#         validation_data=(x_test, x_test)
#     )
#     autoencoders.append(autoencoder)
# 
# # Evaluate and compare the models
# for i, size in enumerate(bottleneck_sizes):
#     loss = autoencoders[i].evaluate(x_test, x_test)
#     print(f'Bottleneck size {size} - Test loss: {loss}')
# ```
# 
# </details>
# 

# ### Exercise 2 - Adding Regularization to the Autoencoder 
#  
# #### Objective: 
# 
# To explore the effect of regularization on the performance of the autoencoder. 
# 
# #### Instructions: 
# 
# **1. Modify the model:**
# - Add L2 regularization to the Dense layers in both the encoder and decoder parts of the autoencoder. 
# 
# **2. Train the model:**
# - Train the modified autoencoder on the MNIST dataset for 50 epochs with a batch size of 256. 
# 
# **3. Evaluate and compare:**
# - Evaluate the performance of the regularized autoencoder and compare it with the non-regularized version. 
# 

# In[14]:


# Write your code here
from tensorflow.keras.regularizers import l2 

# Encoder with L2 regularization 
input_layer = Input(shape=(784,)) 
encoded = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(input_layer) 
bottleneck = Dense(32, activation='relu', kernel_regularizer=l2(0.01))(encoded) 

# Decoder with L2 regularization 
decoded = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(bottleneck) 
output_layer = Dense(784, activation='sigmoid', kernel_regularizer=l2(0.01))(decoded) 

# Autoencoder model with L2 regularization 
autoencoder_regularized = Model(input_layer, output_layer) 
autoencoder_regularized.compile(optimizer='adam', loss='binary_crossentropy') 

# Train the model 
autoencoder_regularized.fit(x_train, x_train, epochs=50, batch_size=256, shuffle=True, validation_data=(x_test, x_test)) 

# Evaluate the model 
loss = autoencoder_regularized.evaluate(x_test, x_test) 
print(f'Regularized Autoencoder - Test loss: {loss}') 


# <details>
#     <summary>Click here for Solution</summary>
# 
# ```python
# 
# from tensorflow.keras.regularizers import l2 
# 
# # Encoder with L2 regularization 
# input_layer = Input(shape=(784,)) 
# encoded = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(input_layer) 
# bottleneck = Dense(32, activation='relu', kernel_regularizer=l2(0.01))(encoded) 
# 
# # Decoder with L2 regularization 
# decoded = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(bottleneck) 
# output_layer = Dense(784, activation='sigmoid', kernel_regularizer=l2(0.01))(decoded) 
# 
# # Autoencoder model with L2 regularization 
# autoencoder_regularized = Model(input_layer, output_layer) 
# autoencoder_regularized.compile(optimizer='adam', loss='binary_crossentropy') 
# 
# # Train the model 
# autoencoder_regularized.fit(x_train, x_train, epochs=50, batch_size=256, shuffle=True, validation_data=(x_test, x_test)) 
# 
# # Evaluate the model 
# loss = autoencoder_regularized.evaluate(x_test, x_test) 
# print(f'Regularized Autoencoder - Test loss: {loss}') 
# ```
# 
# </details>
# 

# ### Exercise 3 - Visualizing Intermediate Representations 
# 
# #### Objective: 
# 
# To visualize and understand the intermediate representations (encoded features) learned by the autoencoder. 
# 
# #### Instructions: 
# 
# **1. Extract Encoder Part:**
# - Extract the encoder part of the trained autoencoder to create a separate model that outputs the encoded features. 
# 
# **2. Visualize Encoded Features:**
# - Use the encoder model to transform the test data into the encoded space. 
# - Plot the encoded features using a scatter plot for the first two dimensions of the encoded space. 
# 

# In[15]:


# Writw your code here
import matplotlib.pyplot as plt 

# Extract the encoder part of the autoencoder 
encoder_model = Model(input_layer, bottleneck) 

# Encode the test data 
encoded_imgs = encoder_model.predict(x_test) 

# Visualize the first two dimensions of the encoded features 
plt.figure(figsize=(10, 8)) 
plt.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], c='blue', alpha=0.5) 
plt.title('Encoded Features - First Two Dimensions') 
plt.xlabel('Encoded Feature 1') 
plt.ylabel('Encoded Feature 2') 
plt.show()


# <details>
#     <summary>Click here for Solution</summary>
# 
# ```python
# 
# import matplotlib.pyplot as plt 
# 
# # Extract the encoder part of the autoencoder 
# encoder_model = Model(input_layer, bottleneck) 
# 
# # Encode the test data 
# encoded_imgs = encoder_model.predict(x_test) 
# 
# # Visualize the first two dimensions of the encoded features 
# plt.figure(figsize=(10, 8)) 
# plt.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], c='blue', alpha=0.5) 
# plt.title('Encoded Features - First Two Dimensions') 
# plt.xlabel('Encoded Feature 1') 
# plt.ylabel('Encoded Feature 2') 
# plt.show() 
# ```
# 
# </details>
# 

# #### Conclusion: 
# 
# Congratulations on completing this lab! In this lab, you have gained practical experience in building, training, and evaluating autoencoders using Keras. You have learned to preprocess data, construct a basic autoencoder architecture, train the model on the MNIST dataset, and visualize the results. Additionally, you explored fine-tuning techniques to enhance the model's performance and applied the autoencoder to denoise images. 
# 
# Continue experimenting with different architectures, datasets, and applications to further deepen your knowledge and skills in using autoencoders. The concepts and techniques you have learned in this lab will serve as a foundation for more advanced topics in deep learning. 
# 

# ## Authors
# 

# Skills Network
# 

# Copyright © IBM Corporation. All rights reserved.
# 
