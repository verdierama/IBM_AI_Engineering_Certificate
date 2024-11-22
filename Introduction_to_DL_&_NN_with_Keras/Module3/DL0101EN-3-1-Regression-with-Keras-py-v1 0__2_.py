#!/usr/bin/env python
# coding: utf-8

# <a href="https://cognitiveclass.ai"><img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DL0101EN-SkillsNetwork/images/IDSN-logo.png" width="400"> </a>
# 
# <h1 align=center><font size = 5>Regression Models with Keras</font></h1>
# 

# ## Introduction
# 

# As we discussed in the videos, despite the popularity of more powerful libraries such as PyToch and TensorFlow, they are not easy to use and have a steep learning curve. So, for people who are just starting to learn deep learning, there is no better library to use other than the Keras library. 
# 
# Keras is a high-level API for building deep learning models. It has gained favor for its ease of use and syntactic simplicity facilitating fast development. As you will see in this lab and the other labs in this course, building a very complex deep learning network can be achieved with Keras with only few lines of code. You will appreciate Keras even more, once you learn how to build deep models using PyTorch and TensorFlow in the other courses.
# 
# So, in this lab, you will learn how to use the Keras library to build a regression model.
# 

# <h2>Regression Models with Keras</h2>
# 
# <h3>Objective for this Notebook<h3>    
# <h5> 1. How to use the Keras library to build a regression model.</h5>
# <h5> 2. Download and Clean dataset </h5>
# <h5> 3. Build a Neural Network </h5>
# <h5> 4. Train and Test the Network. </h5>     
# 
# 

# ## Table of Contents
# 
# <div class="alert alert-block alert-info" style="margin-top: 20px">
# 
# <font size = 3>
#     
# 1. <a href="#item31">Download and Clean Dataset</a>  
# 2. <a href="#item32">Import Keras</a>  
# 3. <a href="#item33">Build a Neural Network</a>  
# 4. <a href="#item34">Train and Test the Network</a>  
# 
# </font>
# </div>
# 

# <a id="item31"></a>
# 

# ## Download and Clean Dataset
# 

# Let's start by importing the <em>pandas</em> and the Numpy libraries.
# 

# In[1]:


# All Libraries required for this lab are listed below. The libraries pre-installed on Skills Network Labs are commented. 
# If you run this notebook on a different environment, e.g. your desktop, you may need to uncomment and install certain libraries.

#!pip install numpy==1.21.4
#!pip install pandas==1.3.4
#!pip install keras==2.1.6


# In[2]:


import pandas as pd
import numpy as np

import warnings
warnings.simplefilter('ignore', FutureWarning)


# We will be playing around with the same dataset that we used in the videos.
# 
# <strong>The dataset is about the compressive strength of different samples of concrete based on the volumes of the different ingredients that were used to make them. Ingredients include:</strong>
# 
# <strong>1. Cement</strong>
# 
# <strong>2. Blast Furnace Slag</strong>
# 
# <strong>3. Fly Ash</strong>
# 
# <strong>4. Water</strong>
# 
# <strong>5. Superplasticizer</strong>
# 
# <strong>6. Coarse Aggregate</strong>
# 
# <strong>7. Fine Aggregate</strong>
# 

# Let's download the data and read it into a <em>pandas</em> dataframe.
# 

# In[3]:


concrete_data = pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0101EN/labs/data/concrete_data.csv')
concrete_data.head()


# So the first concrete sample has 540 cubic meter of cement, 0 cubic meter of blast furnace slag, 0 cubic meter of fly ash, 162 cubic meter of water, 2.5 cubic meter of superplaticizer, 1040 cubic meter of coarse aggregate, 676 cubic meter of fine aggregate. Such a concrete mix which is 28 days old, has a compressive strength of 79.99 MPa. 
# 

# #### Let's check how many data points we have.
# 

# In[4]:


concrete_data.shape


# So, there are approximately 1000 samples to train our model on. Because of the few samples, we have to be careful not to overfit the training data.
# 

# Let's check the dataset for any missing values.
# 

# In[5]:


concrete_data.describe()


# In[6]:


concrete_data.isnull().sum()


# The data looks very clean and is ready to be used to build our model.
# 

# #### Split data into predictors and target
# 

# The target variable in this problem is the concrete sample strength. Therefore, our predictors will be all the other columns.
# 

# In[7]:


concrete_data_columns = concrete_data.columns

predictors = concrete_data[concrete_data_columns[concrete_data_columns != 'Strength']] # all columns except Strength
target = concrete_data['Strength'] # Strength column


# <a id="item2"></a>
# 

# Let's do a quick sanity check of the predictors and the target dataframes.
# 

# In[8]:


predictors.head()


# In[9]:


target.head()


# Finally, the last step is to normalize the data by substracting the mean and dividing by the standard deviation.
# 

# In[10]:


predictors_norm = (predictors - predictors.mean()) / predictors.std()
predictors_norm.head()


# Let's save the number of predictors to *n_cols* since we will need this number when building our network.
# 

# In[11]:


n_cols = predictors_norm.shape[1] # number of predictors


# <a id="item1"></a>
# 

# <a id='item32'></a>
# 

# ## Import Keras
# 

# Recall from the videos that Keras normally runs on top of a low-level library such as TensorFlow. This means that to be able to use the Keras library, you will have to install TensorFlow first and when you import the Keras library, it will be explicitly displayed what backend was used to install the Keras library. In CC Labs, we used TensorFlow as the backend to install Keras, so it should clearly print that when we import Keras.
# 

# #### Let's go ahead and import the Keras library
# 

# In[12]:


import keras


# As you can see, the TensorFlow backend was used to install the Keras library.
# 

# Let's import the rest of the packages from the Keras library that we will need to build our regressoin model.
# 

# In[13]:


from keras.models import Sequential
from keras.layers import Dense


# <a id='item33'></a>
# 

# ## Build a Neural Network
# 

# Let's define a function that defines our regression model for us so that we can conveniently call it to create our model.
# 

# In[14]:


# define regression model
def regression_model():
    # create model
    model = Sequential()
    model.add(Dense(50, activation='relu', input_shape=(n_cols,)))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    
    # compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


# The above function create a model that has two hidden layers, each of 50 hidden units.
# 

# <a id="item4"></a>
# 

# <a id='item34'></a>
# 

# ## Train and Test the Network
# 

# Let's call the function now to create our model.
# 

# In[15]:


# build the model
model = regression_model()


# Next, we will train and test the model at the same time using the *fit* method. We will leave out 30% of the data for validation and we will train the model for 100 epochs.
# 

# In[16]:


# fit the model
model.fit(predictors_norm, target, validation_split=0.3, epochs=100, verbose=2)


# <strong>You can refer to this [link](https://keras.io/models/sequential/) to learn about other functions that you can use for prediction or evaluation.</strong>
# 

# Feel free to vary the following and note what impact each change has on the model's performance:
# 
# 1. Increase or decreate number of neurons in hidden layers
# 2. Add more hidden layers
# 3. Increase number of epochs
# 

# ### Thank you for completing this lab!
# 
# This notebook was created by [Alex Aklson](https://www.linkedin.com/in/aklson/). I hope you found this lab interesting and educational. Feel free to contact me if you have any questions!
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

# This notebook is part of a course on **Coursera** called *Introduction to Deep Learning & Neural Networks with Keras*. If you accessed this notebook outside the course, you can take this course online by clicking [here](https://cocl.us/DL0101EN_Coursera_Week3_LAB1).
# 

# <hr>
# 
# Copyright &copy; 2019 [IBM Developer Skills Network](https://cognitiveclass.ai/?utm_source=bducopyrightlink&utm_medium=dswb&utm_campaign=bdu). This notebook and its source code are released under the terms of the [MIT License](https://bigdatauniversity.com/mit-license/).
# 
