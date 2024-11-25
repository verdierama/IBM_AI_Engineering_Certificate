#jupyter nbconvert --to script "DL0101EN-1-1-Forward-Propgation-py-v1 0__1__.ipynb"
# #!/usr/bin/env python
# coding: utf-8

# <a href="https://cognitiveclass.ai/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkDL0101ENSkillsNetwork945-2022-01-01"><img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DL0101EN-SkillsNetwork/images/IDSN-logo.png" width="400"> </a>
# 
# <h1 align=center><font size = 5>Artificial Neural Networks - Forward Propagation</font></h1>
# 

# ## Introduction
# 

# In this lab, we will build a neural network from scratch and code how it performs predictions using forward propagation. Please note that all deep learning libraries have the entire training and prediction processes implemented, and so in practice you wouldn't really need to build a neural network from scratch. However, hopefully completing this lab will help you understand neural networks and how they work even better.
# 

# <h2>Artificial Neural Networks - Forward Propagation</h2>
# 
# <h3>Objective for this Notebook<h3>    
# <h5> 1. Initalize a Network</h5>
# <h5> 2. Compute Weighted Sum at Each Node. </h5>
# <h5> 3. Compute Node Activation </h5>
# <h5> 4. Access your <b>Flask</b> app via a webpage anywhere using a custom link. </h5>     
# 
# 

# ## Table of Contents
# 
# <div class="alert alert-block alert-info" style="margin-top: 20px">
# 
# <font size = 3>    
# 
# 1. <a href="#item11">Recap</a>
# 2. <a href="#item12">Initalize a Network</a>  
# 3. <a href="#item13">Compute Weighted Sum at Each Node</a>  
# 4. <a href="#item14">Compute Node Activation</a>  
# 5. <a href="#item15">Forward Propagation</a>
# 
# </font>
#     
# </div>
# 

#    
# 

# <a id="item1"></a>
# 

# <a id='item11'></a>
# 

# # Recap
# 

# From the videos, let's recap how a neural network makes predictions through the forward propagation process. Here is a neural network that takes two inputs, has one hidden layer with two nodes, and an output layer with one node.
# 

#    
# 

# <img src="http://cocl.us/neural_network_example" alt="Neural Network Example" width="600px">
# 

#   
# 

# Let's start by randomly initializing the weights and the biases in the network. We have 6 weights and 3 biases, one for each node in the hidden layer as well as for each node in the output layer.
# 

# In[1]:


# All Libraries required for this lab are listed below. The libraries pre-installed on Skills Network Labs are commented. 
# If you run this notebook on a different environment, e.g. your desktop, you may need to uncomment and install certain libraries.

#!pip install numpy==1.21.4


# In[2]:


import numpy as np # import Numpy library to generate 

weights = np.around(np.random.uniform(size=6), decimals=2) # initialize the weights
biases = np.around(np.random.uniform(size=3), decimals=2) # initialize the biases


# Let's print the weights and biases for sanity check.
# 

# In[3]:


print(weights)
print(biases)


# Now that we have the weights and the biases defined for the network, let's compute the output for a given input, $x_1$ and $x_2$.
# 

# In[4]:


x_1 = 0.5 # input 1
x_2 = 0.85 # input 2

print('x1 is {} and x2 is {}'.format(x_1, x_2))


# Let's start by computing the wighted sum of the inputs, $z_{1, 1}$, at the first node of the hidden layer.
# 

# In[7]:


z_11 = x_1 * weights[0] + x_2 * weights[1] + biases[0]

print('The weighted sum of the inputs at the first node in the hidden layer is {}'.format(z_11))


# Next, let's compute the weighted sum of the inputs, $z_{1, 2}$, at the second node of the hidden layer. Assign the value to **z_12**.
# 

# In[8]:


### type your answer here
z_12 = x_1 * weights[2] + x_2 * weights[3] + biases[1]


# Double-click __here__ for the solution.
# <!-- The correct answer is:
# z_12 = x_1 * weights[2] + x_2 * weights[3] + biases[1]
# -->
# 

# Print the weighted sum.
# 

# In[9]:


print('The weighted sum of the inputs at the second node in the hidden layer is {}'.format(np.around(z_12, decimals=4)))


# Next, assuming a sigmoid activation function, let's compute the activation of the first node, $a_{1, 1}$, in the hidden layer.
# 

# In[10]:


a_11 = 1.0 / (1.0 + np.exp(-z_11))

print('The activation of the first node in the hidden layer is {}'.format(np.around(a_11, decimals=4)))


# Let's also compute the activation of the second node, $a_{1, 2}$, in the hidden layer. Assign the value to **a_12**.
# 

# In[11]:


### type your answer here
a_12 = 1.0 / (1.0 + np.exp(-z_12))


# Double-click __here__ for the solution.
# <!-- The correct answer is:
# a_12 = 1.0 / (1.0 + np.exp(-z_12))
# -->
# 

# Print the activation of the second node.
# 

# In[12]:


print('The activation of the second node in the hidden layer is {}'.format(np.around(a_12, decimals=4)))


# Now these activations will serve as the inputs to the output layer. So, let's compute the weighted sum of these inputs to the node in the output layer. Assign the value to **z_2**.
# 

# In[14]:


### type your answer here

z_2 = a_11 * weights[4] + a_12 * weights[5] + biases[2]


# Double-click __here__ for the solution.
# <!-- The correct answer is:
# z_2 = a_11 * weights[4] + a_12 * weights[5] + biases[2]
# -->
# 

# Print the weighted sum of the inputs at the node in the output layer.
# 

# In[15]:


print('The weighted sum of the inputs at the node in the output layer is {}'.format(np.around(z_2, decimals=4)))


# Finally, let's compute the output of the network as the activation of the node in the output layer. Assign the value to **a_2**.
# 

# In[16]:


### type your answer here
a_2 = 1.0 / (1.0 + np.exp(-z_2))


# Double-click __here__ for the solution.
# <!-- The correct answer is:
# a_2 = 1.0 / (1.0 + np.exp(-z_2))
# -->
# 

# Print the activation of the node in the output layer which is equivalent to the prediction made by the network.
# 

# In[17]:


print('The output of the network for x1 = 0.5 and x2 = 0.85 is {}'.format(np.around(a_2, decimals=4)))


# <hr>
# 

# Obviously, neural networks for real problems are composed of many hidden layers and many more nodes in each layer. So, we can't continue making predictions using this very inefficient approach of computing the weighted sum at each node and the activation of each node manually. 
# 

# In order to code an automatic way of making predictions, let's generalize our network. A general network would take $n$ inputs, would have many hidden layers, each hidden layer having $m$ nodes, and would have an output layer. Although the network is showing one hidden layer, but we will code the network to have many hidden layers. Similarly, although the network shows an output layer with one node, we will code the network to have more than one node in the output layer.
# 

# <img src="http://cocl.us/general_neural_network" alt="Neural Network General" width="600px">
# 

#   
# 

# <a id="item2"></a>
# 

# <a id='item12'></a>
# 

# ## Initialize a Network
# 

# Let's start by formally defining the structure of the network.
# 

# In[18]:


n = 2 # number of inputs
num_hidden_layers = 2 # number of hidden layers
m = [2, 2] # number of nodes in each hidden layer
num_nodes_output = 1 # number of nodes in the output layer


# Now that we defined the structure of the network, let's go ahead and inititailize the weights and the biases in the network to random numbers. In order to be able to initialize the weights and the biases to random numbers, we will need to import the **Numpy** library.
# 

# In[19]:


import numpy as np # import the Numpy library

num_nodes_previous = n # number of nodes in the previous layer

network = {} # initialize network an an empty dictionary

# loop through each layer and randomly initialize the weights and biases associated with each node
# notice how we are adding 1 to the number of hidden layers in order to include the output layer
for layer in range(num_hidden_layers + 1): 
    
    # determine name of layer
    if layer == num_hidden_layers:
        layer_name = 'output'
        num_nodes = num_nodes_output
    else:
        layer_name = 'layer_{}'.format(layer + 1)
        num_nodes = m[layer]
    
    # initialize weights and biases associated with each node in the current layer
    network[layer_name] = {}
    for node in range(num_nodes):
        node_name = 'node_{}'.format(node+1)
        network[layer_name][node_name] = {
            'weights': np.around(np.random.uniform(size=num_nodes_previous), decimals=2),
            'bias': np.around(np.random.uniform(size=1), decimals=2),
        }
    
    num_nodes_previous = num_nodes
    
print(network) # print network


# Awesome! So now with the above code, we are able to initialize the weights and the biases pertaining to any network of any number of hidden layers and number of nodes in each layer. But let's put this code in a function so that we are able to repetitively execute all this code whenever we want to construct a neural network.
# 

# In[20]:


def initialize_network(num_inputs, num_hidden_layers, num_nodes_hidden, num_nodes_output):
    
    num_nodes_previous = num_inputs # number of nodes in the previous layer

    network = {}
    
    # loop through each layer and randomly initialize the weights and biases associated with each layer
    for layer in range(num_hidden_layers + 1):
        
        if layer == num_hidden_layers:
            layer_name = 'output' # name last layer in the network output
            num_nodes = num_nodes_output
        else:
            layer_name = 'layer_{}'.format(layer + 1) # otherwise give the layer a number
            num_nodes = num_nodes_hidden[layer] 
        
        # initialize weights and bias for each node
        network[layer_name] = {}
        for node in range(num_nodes):
            node_name = 'node_{}'.format(node+1)
            network[layer_name][node_name] = {
                'weights': np.around(np.random.uniform(size=num_nodes_previous), decimals=2),
                'bias': np.around(np.random.uniform(size=1), decimals=2),
            }
    
        num_nodes_previous = num_nodes

    return network # return the network


# #### Use the *initialize_network* function to create a network that:
# 
# 1. takes 5 inputs
# 2. has three hidden layers
# 3. has 3 nodes in the first layer, 2 nodes in the second layer, and 3 nodes in the third layer
# 4. has 1 node in the output layer
# 
# Call the network **small_network**.
# 

# In[25]:


### type your answer here
small_network = initialize_network(num_inputs=5, num_hidden_layers=3, num_nodes_hidden = [3, 2, 3], num_nodes_output=1)
small_network


# Double-click __here__ for the solution.
# <!-- The correct answer is:
# small_network = initialize_network(5, 3, [3, 2, 3], 1)
# -->
# 

#   
# 

# <a id="item3"></a>
# 

# <a id='item13'></a>
# 

# ## Compute Weighted Sum at Each Node
# 

# The weighted sum at each node is computed as the dot product of the inputs and the weights plus the bias. So let's create a function called *compute_weighted_sum* that does just that.
# 

# In[26]:


def compute_weighted_sum(inputs, weights, bias):
    return np.sum(inputs * weights) + bias


# Let's generate 5 inputs that we can feed to **small_network**.
# 

# In[28]:


from random import seed
import numpy as np

np.random.seed(12)
inputs = np.around(np.random.uniform(size=5), decimals=2)

print('The inputs to the network are {}'.format(inputs))


# #### Use the *compute_weighted_sum* function to compute the weighted sum at the first node in the first hidden layer.
# 

# In[31]:


### type your answer here
weighted_sum = compute_weighted_sum(inputs, small_network['layer_1']['node_1']['weights'], small_network['layer_1']['node_1']['bias'])
weighted_sum



# Double-click __here__ for the solution.
# <!-- The correct answer is:
# node_weights = small_network['layer_1']['node_1']['weights']
# node_bias = small_network['layer_1']['node_1']['bias']
# 
# weighted_sum = compute_weighted_sum(inputs, node_weights, node_bias)
# print('The weighted sum at the first node in the hidden layer is {}'.format(np.around(weighted_sum[0], decimals=4)))
# -->
# 

#    
# 

# <a id="item4"></a>
# 

# <a id='item14'></a>
# 

# ## Compute Node Activation
# 

# Recall that the output of each node is simply a non-linear tranformation of the weighted sum. We use activation functions for this mapping. Let's use the sigmoid function as the activation function here. So let's define a function that takes a weighted sum as input and returns the non-linear transformation of the input using the sigmoid function.
# 

# In[32]:


def node_activation(weighted_sum):
    return 1.0 / (1.0 + np.exp(-1 * weighted_sum))


# #### Use the *node_activation* function to compute the output of the first node in the first hidden layer.
# 

# In[38]:


### type your answer here
output = node_activation(weighted_sum)
format(np.around(output[0], decimals=4))


# Double-click __here__ for the solution.
# <!-- The correct answer is:
# node_output  = node_activation(compute_weighted_sum(inputs, node_weights, node_bias))
# print('The output of the first node in the hidden layer is {}'.format(np.around(node_output[0], decimals=4)))
# -->
# 

#    
# 

# <a id="item5"></a>
# 

# <a id='item15'></a>
# 

# ## Forward Propagation
# 

# The final piece of building a neural network that can perform predictions is to put everything together. So let's create a function that applies the *compute_weighted_sum* and *node_activation* functions to each node in the network and propagates the data all the way to the output layer and outputs a prediction for each node in the output layer.
# 

# The way we are going to accomplish this is through the following procedure:
# 
# 1. Start with the input layer as the input to the first hidden layer.
# 2. Compute the weighted sum at the nodes of the current layer.
# 3. Compute the output of the nodes of the current layer.
# 4. Set the output of the current layer to be the input to the next layer.
# 5. Move to the next layer in the network.
# 5. Repeat steps 2 - 4 until we compute the output of the output layer.
# 

# In[40]:


def forward_propagate(network, inputs):
    
    layer_inputs = list(inputs) # start with the input layer as the input to the first hidden layer
    
    for layer in network:
        
        layer_data = network[layer]
        
        layer_outputs = [] 
        for layer_node in layer_data:
        
            node_data = layer_data[layer_node]
        
            # compute the weighted sum and the output of each node at the same time 
            node_output = node_activation(compute_weighted_sum(layer_inputs, node_data['weights'], node_data['bias']))
            layer_outputs.append(np.around(node_output[0], decimals=4))
            
        if layer != 'output':
            print('The outputs of the nodes in hidden layer number {} is {}'.format(layer.split('_')[1], layer_outputs))
    
        layer_inputs = layer_outputs # set the output of this layer to be the input to next layer

    network_predictions = layer_outputs
    return network_predictions


# #### Use the *forward_propagate* function to compute the prediction of our small network
# 

# In[43]:


### type your answser here
predictions = forward_propagate(small_network, inputs)
print('The predicted value by the network for the given input is {}'.format(np.around(predictions[0], decimals=4)))



# Double-click __here__ for the solution.
# <!-- The correct answer is:
# predictions = forward_propagate(small_network, inputs)
# print('The predicted value by the network for the given input is {}'.format(np.around(predictions[0], decimals=4)))
# -->
# 

# So we built the code to define a neural network. We can specify the number of inputs that a neural network can take, the number of hidden layers as well as the number of nodes in each hidden layer, and the number of nodes in the output layer.
# 

# We first use the *initialize_network* to create our neural network and define its weights and biases.
# 

# In[44]:


my_network = initialize_network(5, 3, [2, 3, 2], 3)


# Then, for a given input,
# 

# In[45]:


inputs = np.around(np.random.uniform(size=5), decimals=2)


# we compute the network predictions.
# 

# In[46]:


predictions = forward_propagate(my_network, inputs)
print('The predicted values by the network for the given input are {}'.format(predictions))


# Feel free to play around with the code by creating different networks of different structures and enjoy making predictions using the *forward_propagate* function.
# 

# In[47]:


### create a network
my_network2 = initialize_network(15, 4, [12, 13, 12, 5], 13)
inputs = np.around(np.random.uniform(size=5), decimals=2)
predictions = forward_propagate(my_network, inputs)
print('The predicted values by the network for the given input are {}'.format(predictions))



# In[ ]:


### create another network





# ### Thank you for completing this lab!
# 
# This notebook was created by [Alex Aklson](https://www.linkedin.com/in/aklson/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkDL0101ENSkillsNetwork945-2022-01-01). I hope you found this lab interesting and educational. Feel free to contact me if you have any questions!
# 

# 
# ## Change Log
# 
# |  Date (YYYY-MM-DD) |  Version | Changed By  |  Change Description |
# |---|---|---|---|
# | 2020-09-21  | 2.0  | Srishti  |  Migrated Lab to Markdown and added to course repo in GitLab |
# 
# 
# 
# <hr>
# 
# ## <h3 align="center"> © IBM Corporation 2020. All rights reserved. <h3/>
# 

# This notebook is part of a course on **Coursera** called *Introduction to Deep Learning & Neural Networks with Keras*. If you accessed this notebook outside the course, you can take this course online by clicking [here](https://cocl.us/DL0101EN_Coursera_Week1_LAB1).
# 

# <hr>
# 
# Copyright &copy; 2019 [IBM Developer Skills Network](https://cognitiveclass.ai/?utm_medium=dswb&utm_source=bducopyrightlink&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkDL0101ENSkillsNetwork945-2022-01-01&utm_campaign=bdu). This notebook and its source code are released under the terms of the [MIT License](https://bigdatauniversity.com/mit-license/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkDL0101ENSkillsNetwork945-2022-01-01).
# 
