#!/usr/bin/env python
# coding: utf-8

# <p style="text-align:center">
#     <a href="https://skills.network" target="_blank">
#     <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/assets/logos/SN_web_lightmode.png" width="200" alt="Skills Network Logo">
#     </a>
# </p>
# 

# <h1>Multiple Linear Regression</h1>
# 

# <h2>Objective</h2><ul><li> How to make the prediction for multiple inputs.</li><li> How to use linear class to build more complex models.</li><li> How to build a custom module.</li></ul> 
# 

# <h2>Table of Contents</h2>
# <p>In this lab, you will review how to make a prediction in several different ways by using PyTorch.</p>
#  
# <div class="alert alert-block alert-info" style="margin-top: 20px">
# <font size="3">
#         1. <a href="#Prediction">Prediction</a><br>
#         2. <a href="#Class-Linear">Class Linear</a><br>
#         3. <a href="#Build-Custom-Modules">Build Custom Modules</a>
#     
# <p>Estimated Time Needed: <strong>15 min</strong></p>
# </font>
# </div>
# 

# <h2>Preparation</h2>
# 

# Import the libraries and set the random seed.
# 

# In[1]:


# Import the libraries and set the random seed

from torch import nn
import torch
torch.manual_seed(1)


# <!--Empty Space for separating topics-->
# 

# <h2 id="Prediction">Prediction</h2>
# 

# Set weight and bias.
# 

# In[2]:


# Set the weight and bias

w = torch.tensor([[2.0], [3.0]], requires_grad=True)
b = torch.tensor([[1.0]], requires_grad=True)


# Define the parameters. <code>torch.mm</code> uses matrix multiplication instead of scaler multiplication.
# 

# In[3]:


# Define Prediction Function

def forward(x):
    yhat = torch.mm(x, w) + b
    return yhat


# The function <code>forward</code> implements the following equation:
# 

# <img src="https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0110EN/notebook_images%20/chapter2/2.6.1_matrix_eq.png" width="600" alt="Matrix Linear Regression">
# 

# If we input a <i>1x2</i> tensor, because we have a <i>2x1</i> tensor as <code>w</code>, we will get a <i>1x1</i> tensor: 
# 

# In[4]:


# Calculate yhat

x = torch.tensor([[1.0, 2.0]])
yhat = forward(x)
print("The result: ", yhat)


# <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/JBkvPoMCCa-PDXCF_4aQfQ/image%20-1-.png" width="300" alt="Linear Regression Matrix Sample One">
# 

# # Each row of the following tensor represents a sample:
# 

# In[5]:


# Sample tensor X

X = torch.tensor([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])


# In[6]:


# Make the prediction of X 

yhat = forward(X)
print("The result: ", yhat)


# <!--Empty Space for separating topics-->
# 

# ## Class Linear
# 

# We can use the linear class to make a prediction. You'll also use the linear class to build more complex models.
# 

# Let us create a model.
# 

# In[7]:


# Make a linear regression model using build-in function

model = nn.Linear(2, 1)


# Make a prediction with the first sample:
# 

# In[8]:


# Make a prediction of x

yhat = model(x)
print("The result: ", yhat)


# Predict with multiple samples <code>X</code>: 
# 

# In[9]:


# Make a prediction of X

yhat = model(X)
print("The result: ", yhat)


# The function performs matrix multiplication as shown in this image:
# 

# <img src="https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0110EN/notebook_images%20/chapter2/2.6.1multi_sample_example.png" width="600" alt="Linear Regression Matrix Sample One">
# 

# <!--Empty Space for separating topics-->
# 

# ## Build Custom Modules
# 

# Now, you'll build a custom module. You can make more complex models by using this method later. 
# 

# In[10]:


# Create linear_regression Class

class linear_regression(nn.Module):
    
    # Constructor
    def __init__(self, input_size, output_size):
        super(linear_regression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
    
    # Prediction function
    def forward(self, x):
        yhat = self.linear(x)
        return yhat


# Build a linear regression object. The input feature size is two. 
# 

# In[11]:


model = linear_regression(2, 1)


# This will input the following equation:
# 

# <img src="https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0110EN/notebook_images%20/chapter2/2.6.1_matrix_eq.png" width="600" alt="Matrix Linear Regression">
# 

# You can see the randomly initialized parameters by using the <code>parameters()</code> method:
# 

# In[12]:


# Print model parameters

print("The parameters: ", list(model.parameters()))


# You can also see the parameters by using the <code>state_dict()</code> method:
# 

# In[13]:


# Print model parameters

print("The parameters: ", model.state_dict())


# Now we input a 1x2 tensor, and we will get a 1x1 tensor.
# 

# In[14]:


# Make a prediction of x

yhat = model(x)
print("The result: ", yhat)


# The shape of the output is shown in the following image: 
# 

# <img src="https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0110EN/notebook_images%20/chapter2/2.6.1_matrix_eq.png" width="600" alt="Matrix Linear Regression">
# 

# Make a prediction for multiple samples:
# 

# In[15]:


# Make a prediction of X

yhat = model(X)
print("The result: ", yhat)


# The shape is shown in the following image: 
# 

# <img src="https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0110EN/notebook_images%20/chapter2/2.6.1Multi_sample.png" width="600" alt="Multiple Samples Linear Regression">
# 

# <!--Empty Space for separating topics-->
# 

# <h3>Practice</h3>
# 

# Build a model or object of type <code>linear_regression</code>. Using the <code>linear_regression</code> object will predict the following tensor: 
# 

# In[19]:


# Practice: Build a model to predict the follow tensor.

X = torch.tensor([[11.0, 12.0, 13, 14], [11, 12, 13, 14]])
model = linear_regression(4, 1)
yhat = model(X)
print("The result: ", yhat)


# Double-click <b>here</b> for the solution.
# <!-- Your answer is below:
# model = linear_regression(4, 1)
# yhat = model(X)
# print("The result: ", yhat)
# -->
# 

# 
# 
# <a href="https://dataplatform.cloud.ibm.com/registration/stepone?utm_source=skills_network&utm_content=in_lab_content_link&utm_id=Lab-IBMDeveloperSkillsNetwork-DL0110EN-SkillsNetwork&context=cpdaas&apps=data_science_experience%2Cwatson_machine_learning"><img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DL0110EN-SkillsNetwork/Template/module%201/images/Watson_Studio.png"></a>
# 

# <!--Empty Space for separating topics-->
# 

# <h2>About the Authors:</h2> 
# 
# <a href="https://www.linkedin.com/in/joseph-s-50398b136/">Joseph Santarcangelo</a> has a PhD in Electrical Engineering, his research focused on using machine learning, signal processing, and computer vision to determine how videos impact human cognition. Joseph has been working for IBM since he completed his PhD.
# 

# Other contributors: <a href="https://www.linkedin.com/in/michelleccarey/">Michelle Carey</a>, <a href="https://www.linkedin.com/in/jiahui-mavis-zhou-a4537814a/">Mavis Zhou</a>
# 

# <!--
# ## Change Log
# 
# |  Date (YYYY-MM-DD) |  Version | Changed By  |  Change Description |
# |---|---|---|---|
# | 2020-09-23  | 2.0  | Shubham  |  Migrated Lab to Markdown and added to course repo in GitLab |
# -->
# 

# <hr>
# 

# ## <h3 align="center"> &#169; IBM Corporation. All rights reserved. <h3/>
# 
