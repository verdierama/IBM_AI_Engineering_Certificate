# jupyter nbconvert --to script "REVIEW-Lab_Building Advanced Transformers.ipynb"
# #!/usr/bin/env python
# coding: utf-8

# <p style="text-align:center">
#     <a href="https://skills.network" target="_blank">
#     <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/assets/logos/SN_web_lightmode.png" width="200" alt="Skills Network Logo"  />
#     </a>
# </p>
# 

# # **Lab: Building Advanced Transformers**
# 
# **Estimated time needed:  30 minutes**  
# 
# In this lab, you will implement and experiment with advanced Transformer models using Keras. 
# 
# **Learning objectives:** 
# 
# By the end of this lab, you will: 
# 
# - Implement advanced Transformer models using Keras. 
# 
# - Apply Transformers to real-world sequential data tasks. 
# 
# - Build, train, and evaluate Transformer models. 
# 

# ## Step-by-Step Instructions: 
# 
# ### Step 1: Import necessary libraries 
# 
# Before you start, you need to import the required libraries: TensorFlow and Keras. Keras is included within TensorFlow as `tensorflow.keras.`
# 

# In[1]:


#get_ipython().run_line_magic('pip', 'install tensorflow pyarrow')
#get_ipython().run_line_magic('pip', 'install pandas')
#get_ipython().run_line_magic('pip', 'install scikit-learn')
#get_ipython().run_line_magic('pip', 'install matplotlib')
#get_ipython().run_line_magic('pip', 'install requests')



# In[2]:


import numpy as np 
import pandas as pd 
import tensorflow as tf 
import requests
from sklearn.preprocessing import MinMaxScaler 
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, Dropout


# ####  Setup the Environment to generate synthetic stock price data
# 

# In[3]:


import numpy as np
import pandas as pd

# Create a synthetic stock price dataset
np.random.seed(42)
data_length = 2000  # Adjust data length as needed
trend = np.linspace(100, 200, data_length)
noise = np.random.normal(0, 2, data_length)
synthetic_data = trend + noise

# Create a DataFrame and save as 'stock_prices.csv'
data = pd.DataFrame(synthetic_data, columns=['Close'])
data.to_csv('stock_prices.csv', index=False)
print("Synthetic stock_prices.csv created and loaded.")


# In[4]:


# Load the dataset 
data = pd.read_csv('stock_prices.csv') 
data = data[['Close']].values 

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
data = scaler.fit_transform(data)

# Prepare the data for training
def create_dataset(data, time_step=1):
    X, Y = [], []

    for i in range(len(data)-time_step-1):
        a = data[i:(i+time_step), 0]
        X.append(a)
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

time_step = 100
X, Y = create_dataset(data, time_step)
X = X.reshape(X.shape[0], X.shape[1], 1)

print("Shape of X:", X.shape) 
print("Shape of Y:", Y.shape) 


# In the above code: 
# 
# `tensorflow` is the main library for machine learning in Python.  
# 
# `stock_prices.csv` is the data set that is loaded. 
# 
# `MinMaxScaler` method is used to normalize the data.  
# 
# `create_dataset`method is used to prepare the data for training. 
# 

# ### Step 2: Implement Multi-Head Self-Attention 
# 
# Define the Multi-Head Self-Attention mechanism. 
# 

# In[5]:


class MultiHeadSelfAttention(Layer): 

    def __init__(self, embed_dim, num_heads=8): 
        super(MultiHeadSelfAttention, self).__init__() 
        self.embed_dim = embed_dim 
        self.num_heads = num_heads 
        self.projection_dim = embed_dim // num_heads 
        self.query_dense = Dense(embed_dim) 
        self.key_dense = Dense(embed_dim) 
        self.value_dense = Dense(embed_dim) 
        self.combine_heads = Dense(embed_dim) 


    def attention(self, query, key, value): 
        score = tf.matmul(query, key, transpose_b=True) 
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32) 
        scaled_score = score / tf.math.sqrt(dim_key) 
        weights = tf.nn.softmax(scaled_score, axis=-1) 
        output = tf.matmul(weights, value) 
        return output, weights 

    def split_heads(self, x, batch_size): 
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim)) 
        return tf.transpose(x, perm=[0, 2, 1, 3]) 

    def call(self, inputs): 
        batch_size = tf.shape(inputs)[0] 
        query = self.query_dense(inputs) 
        key = self.key_dense(inputs) 
        value = self.value_dense(inputs) 
        query = self.split_heads(query, batch_size) 
        key = self.split_heads(key, batch_size) 
        value = self.split_heads(value, batch_size) 
        attention, _ = self.attention(query, key, value) 
        attention = tf.transpose(attention, perm=[0, 2, 1, 3]) 
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim)) 
        output = self.combine_heads(concat_attention) 
        return output 

 


# In the above code: 
# 
# - The MultiHeadSelfAttention layer implements the multi-head self-attention mechanism, which allows the model to focus on different parts of the input sequence simultaneously. 
# 
# - The attention parameter computes the attention scores and weighted sum of the values. 
# 
# - The split_heads parameter splits the input into multiple heads for parallel attention computation. 
# 
# - The call method applies the self-attention mechanism and combines the heads. 
# 

# ### Step 3: Implement Transformer block 
# 
# Define the Transformer block. 
# 

# In[6]:


class TransformerBlock(Layer): 

    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1): 
        super(TransformerBlock, self).__init__() 
        self.att = MultiHeadSelfAttention(embed_dim, num_heads) 
        self.ffn = tf.keras.Sequential([ 
            Dense(ff_dim, activation="relu"), 
            Dense(embed_dim), 
        ]) 

        self.layernorm1 = LayerNormalization(epsilon=1e-6) 
        self.layernorm2 = LayerNormalization(epsilon=1e-6) 
        self.dropout1 = Dropout(rate) 
        self.dropout2 = Dropout(rate) 


    def call(self, inputs, training): 
        attn_output = self.att(inputs) 
        attn_output = self.dropout1(attn_output, training=training) 
        out1 = self.layernorm1(inputs + attn_output) 
        ffn_output = self.ffn(out1) 
        ffn_output = self.dropout2(ffn_output, training=training) 
        return self.layernorm2(out1 + ffn_output) 


# In the above code:
# 
# - The TransformerBlock layer combines multi-head self-attention with a feed-forward neural network and normalization layers.  
# 
# - Dropout is used to prevent overfitting. 
# 
# - The call method applies the self-attention, followed by the feedforward network with residual connections and layer normalization.
# 

# ### Step 4: Implement Encoder Layer 
# 
# Define the Encoder layer. 
# 

# In[7]:


class EncoderLayer(Layer): 

    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1): 
        super(EncoderLayer, self).__init__() 
        self.att = MultiHeadSelfAttention(embed_dim, num_heads) 
        self.ffn = tf.keras.Sequential([ 
            Dense(ff_dim, activation="relu"), 
            Dense(embed_dim), 
        ]) 

        self.layernorm1 = LayerNormalization(epsilon=1e-6) 
        self.layernorm2 = LayerNormalization(epsilon=1e-6) 
        self.dropout1 = Dropout(rate) 
        self.dropout2 = Dropout(rate) 

 

    def call(self, inputs, training): 
        attn_output = self.att(inputs) 
        attn_output = self.dropout1(attn_output, training=training) 
        out1 = self.layernorm1(inputs + attn_output) 
        ffn_output = self.ffn(out1) 
        ffn_output = self.dropout2(ffn_output, training=training) 
        return self.layernorm2(out1 + ffn_output) 



# In the above code: 
# 
# - The EncoderLayer is similar to the TransformerBlock but is a reusable layer in the Transformer architecture. 
# 
# - It consists of a MultiHeadSelfAttention mechanism followed by a feedforward neural network. 
# 
# - Both sub-layers have residual connections around them, and layer normalization is applied to the output of each sub-layer. 
# 
# - The call method applies the self-attention, followed by the feedforward network, with residual connections and layer normalization. 
# 

# ### Step 5: Implement Transformer encoder 
# 
# Define the Transformer Encoder. 
# 

# In[8]:


import tensorflow as tf 
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, Dropout 

class MultiHeadSelfAttention(Layer): 
    def __init__(self, embed_dim, num_heads=8): 
        super(MultiHeadSelfAttention, self).__init__() 
        self.embed_dim = embed_dim 
        self.num_heads = num_heads 
        self.projection_dim = embed_dim // num_heads 
        self.query_dense = Dense(embed_dim) 
        self.key_dense = Dense(embed_dim) 
        self.value_dense = Dense(embed_dim) 
        self.combine_heads = Dense(embed_dim) 
 

    def attention(self, query, key, value): 
        score = tf.matmul(query, key, transpose_b=True) 
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32) 
        scaled_score = score / tf.math.sqrt(dim_key) 
        weights = tf.nn.softmax(scaled_score, axis=-1) 
        output = tf.matmul(weights, value) 
        return output, weights 


    def split_heads(self, x, batch_size): 
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim)) 
        return tf.transpose(x, perm=[0, 2, 1, 3]) 


    def call(self, inputs): 
        batch_size = tf.shape(inputs)[0] 
        query = self.query_dense(inputs) 
        key = self.key_dense(inputs) 
        value = self.value_dense(inputs) 
        query = self.split_heads(query, batch_size) 
        key = self.split_heads(key, batch_size) 
        value = self.split_heads(value, batch_size) 
        attention, _ = self.attention(query, key, value) 
        attention = tf.transpose(attention, perm=[0, 2, 1, 3]) 
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim)) 
        output = self.combine_heads(concat_attention) 
        return output 

class TransformerBlock(Layer): 
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1): 
        super(TransformerBlock, self).__init__() 
        self.att = MultiHeadSelfAttention(embed_dim, num_heads) 
        self.ffn = tf.keras.Sequential([ 
            Dense(ff_dim, activation="relu"), 
            Dense(embed_dim), 
        ]) 

        self.layernorm1 = LayerNormalization(epsilon=1e-6) 
        self.layernorm2 = LayerNormalization(epsilon=1e-6) 
        self.dropout1 = Dropout(rate) 
        self.dropout2 = Dropout(rate) 
 

    def call(self, inputs, training): 
        attn_output = self.att(inputs) 
        attn_output = self.dropout1(attn_output, training=training) 
        out1 = self.layernorm1(inputs + attn_output) 
        ffn_output = self.ffn(out1) 
        ffn_output = self.dropout2(ffn_output, training=training) 
        return self.layernorm2(out1 + ffn_output) 

class TransformerEncoder(Layer): 
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, rate=0.1): 
        super(TransformerEncoder, self).__init__() 
        self.num_layers = num_layers 
        self.embed_dim = embed_dim 
        self.enc_layers = [TransformerBlock(embed_dim, num_heads, ff_dim, rate) for _ in range(num_layers)] 
        self.dropout = Dropout(rate) 

    def call(self, inputs, training=False): 
        x = inputs 
        for i in range(self.num_layers): 
            x = self.enc_layers[i](x, training=training) 
        return x 

# Example usage 
embed_dim = 128 
num_heads = 8 
ff_dim = 512 
num_layers = 4 

transformer_encoder = TransformerEncoder(num_layers, embed_dim, num_heads, ff_dim) 
inputs = tf.random.uniform((1, 100, embed_dim)) 
outputs = transformer_encoder(inputs, training=False)  # Use keyword argument for 'training' 
print(outputs.shape)  # Should print (1, 100, 128) 


# In the above code: 
# 
# The TransformerEncoder is composed of multiple TransformerBlock layers, implementing the encoding part of the Transformer architecture. 
# 

# ### Step 6: Build and Compile the Transformer model 
# 
# Integrate the Transformer Encoder into a complete model for sequential data. 
# 

# In[16]:


# Define the necessary parameters 

embed_dim = 128 
num_heads = 8 
ff_dim = 512 
num_layers = 4 

# Define the Transformer Encoder 
transformer_encoder = TransformerEncoder(num_layers, embed_dim, num_heads, ff_dim) 

# Build the model 
input_shape = (X.shape[1], X.shape[2]) 
inputs = tf.keras.Input(shape=input_shape) 

# Project the inputs to the embed_dim 
x = tf.keras.layers.Dense(embed_dim)(inputs) 
encoder_outputs = transformer_encoder(x) 
flatten = tf.keras.layers.Flatten()(encoder_outputs) 
outputs = tf.keras.layers.Dense(1)(flatten) 
model = tf.keras.Model(inputs, outputs) 

# Compile the model 
model.compile(optimizer='adam', loss='mse') 

# Summary of the model 
model.summary() 


# In the above code: 
# 
# - The Transformer Encoder model defines the necessary parameters, flattens the output, and ends with a dense layer to produce the final output.  
# 
# - The model is then compiled with the Adam optimizer and mean squared error loss. 
# 

# ### Step 7: Train the Transformer model 
# 
# Train the model on the prepared dataset. 
# 

# In[10]:


# Train the model
model.fit(X, Y, epochs=20, batch_size=32)


# In the above code: 
# 
# The model is trained on the normalized stock price data for 20 epochs with a batch size of 32. 
# 

# ### Step 8: Evaluate and Make Predictions 
# 
# Evaluate the model's performance and make predictions on the dataset. 
# 

# In[15]:


# Make predictions 
predictions = model.predict(X) 
predictions = scaler.inverse_transform(predictions) 
 

# Plot the predictions 
import matplotlib.pyplot as plt 

#plt.plot(data, label='True Data') 
plt.plot(scaler.inverse_transform(data), label='True Data') 
plt.plot(np.arange(time_step, time_step + len(predictions)), predictions, label='Predictions') 
plt.xlabel('Time') 
plt.ylabel('Stock Price') 
plt.legend() 
plt.show() 

 


# In the above code: 
# 
# - The model's predictions are transformed back to the original scale using the inverse transform of the scaler. 
# 
# - The true data and predictions are plotted to visualize the model's performance. 
# 

# ## Practice Exercises: 
# 
#  ### Exercise 1: Add dropout to the Transformer model 
# 
#  **Objective: Understand how to add dropout layers to the Transformer model to prevent overfitting.** 
# 
#  Instructions: 
# 
# - Add a dropout layer after the Flatten layer in the model. 
# 
# - Set the dropout rate to 0.5. 
# 

# In[17]:


## Write your code here.
# Add a dropout layer after the Flatten layer 

flatten = tf.keras.layers.Flatten()(encoder_outputs) 

dropout = Dropout(0.5)(flatten) 

outputs = tf.keras.layers.Dense(1)(dropout) 

  

# Build the model 

model = tf.keras.Model(inputs, outputs) 

  

# Compile the model 

model.compile(optimizer='adam', loss='mse') 

  

# Train the model 

model.fit(X, Y, epochs=20, batch_size=32) 

  

# Evaluate the model 

loss = model.evaluate(X, Y) 

print(f'Test loss: {loss}') 


# <details><summary>Click here to view the solution.</summary>
# 
# ```
# from tensorflow.keras.layers import Dropout 
# 
#   
# 
# # Add a dropout layer after the Flatten layer 
# 
# flatten = tf.keras.layers.Flatten()(encoder_outputs) 
# 
# dropout = Dropout(0.5)(flatten) 
# 
# outputs = tf.keras.layers.Dense(1)(dropout) 
# 
#   
# 
# # Build the model 
# 
# model = tf.keras.Model(inputs, outputs) 
# 
#   
# 
# # Compile the model 
# 
# model.compile(optimizer='adam', loss='mse') 
# 
#   
# 
# # Train the model 
# 
# model.fit(X, Y, epochs=20, batch_size=32) 
# 
#   
# 
# # Evaluate the model 
# 
# loss = model.evaluate(X, Y) 
# 
# print(f'Test loss: {loss}') 
# 
# ```
# </details>
# 

# ### Exercise 2: Experiment with different batch sizes 
# 
# **Objective: Observe the impact of different batch sizes on model performance.** 
# 
#  Instructions: 
# 
# - Train the model with a batch size of 16. 
# 
# - Train the model with a batch size of 64. 
# 
# - Compare the training time and performance. 
# 

# In[18]:


## Write your code here.

# Train the model 
model.fit(X, Y, epochs=20, batch_size=16) 

# Evaluate the model 
loss = model.evaluate(X, Y) 

print(f'Test loss for batch of 16: {loss}') 

# Train the model 
model.fit(X, Y, epochs=20, batch_size=64) 

# Evaluate the model 
loss = model.evaluate(X, Y) 

print(f'Test loss for batch of 64: {loss}') 


# <details><summary>Click here to view the solution.</summary>
# 
# ```
# # Train the model with batch size 16
# model.fit(X, Y, epochs=20, batch_size=16)
# 
# # Evaluate the model
# loss = model.evaluate(X, Y)
# print(f'Test loss with batch size 16: {loss}')
# 
# # Train the model with batch size 64
# model.fit(X, Y, epochs=20, batch_size=64)
# 
# # Evaluate the model
# loss = model.evaluate(X, Y)
# print(f'Test loss with batch size 64: {loss}')
# 
# ```
# </details>
# 

# ### Exercise 3: Use a different activation function 
# 
#  **Objective: Understand how different activation functions impact the model performance.** 
# 
#  Instructions: 
# 
# - Change the activation function of the Dense layer to `tanh`. 
# 
# - Train and evaluate the model. 
# 

# In[19]:


## Write your code here.
outputs = tf.keras.layers.Dense(1, activation='tanh')(flatten)
# Build the model
model = tf.keras.Model(inputs, outputs)
# Compile the model 
model.compile(optimizer='adam', loss='mse') 
# Train the model 
model.fit(X, Y, epochs=20, batch_size=32) 
# Evaluate the model 
loss = model.evaluate(X, Y) 
print(f'Test loss  witn tanh: {loss}') 


# <details><summary>Click here to view the solution.</summary>
# 
# ```
# # Change the activation function of the Dense layer to tanh
# outputs = tf.keras.layers.Dense(1, activation='tanh')(flatten)
# 
# # Build the model
# model = tf.keras.Model(inputs, outputs)
# 
# # Compile the model
# model.compile(optimizer='adam', loss='mse')
# 
# # Train the model
# model.fit(X, Y, epochs=20, batch_size=32)
# 
# # Evaluate the model
# loss = model.evaluate(X, Y)
# print(f'Test loss with tanh activation: {loss}')
# 
# ```
# </details>
# 

# ## Conclusion
# Congratulations on completing this lab! In this lab, you have built an advanced Transformer model using Keras and applied it to a time series forecasting task. You have learned how to define and implement multi-head self-attention, Transformer blocks, encoder layers, and integrate them into a complete Transformer model. By experimenting with different configurations and training the model, you can further improve its performance and apply it to various sequential data tasks. 
# 

# Copyright © IBM Corporation. All rights reserved.
# 
