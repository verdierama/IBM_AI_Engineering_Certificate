#!/usr/bin/env python
# coding: utf-8

# <a href="https://cognitiveclass.ai"><img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DL0101EN-SkillsNetwork/images/IDSN-logo.png" width="400"> </a>
# 
# <h1 align=center><font size = 5>Regression Models with Keras</font></h1>
# 

# <h2>Regression Models with Keras</h2>
#  

# <a id="item31"></a>
# 

# ## Download and Clean Dataset
# 

# Let's start by importing the <em>pandas</em> and the Numpy libraries.
# 

# In[83]:


# All Libraries required for this lab are listed below. The libraries pre-installed on Skills Network Labs are commented. 
# If you run this notebook on a different environment, e.g. your desktop, you may need to uncomment and install certain libraries.

#!pip install numpy==1.21.4
#!pip install pandas==1.3.4
#!pip install keras==2.1.6


# In[84]:


import pandas as pd
import numpy as np

import warnings
warnings.simplefilter('ignore', FutureWarning)


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

# In[85]:


concrete_data = pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0101EN/labs/data/concrete_data.csv')
concrete_data.head()


# 
# 

# #### Let's check how many data points we have.
# 

# In[86]:


concrete_data.shape


# So, there are approximately 1000 samples to train our model on. Because of the few samples, we have to be careful not to overfit the training data.
# 

# Let's check the dataset for any missing values.
# 

# In[87]:


concrete_data.describe()


# In[88]:


concrete_data.isnull().sum()


# The data looks very clean and is ready to be used to build our model.
# 

# #### Split data into predictors and target
# 

# The target variable in this problem is the concrete sample strength. Therefore, our predictors will be all the other columns.
# 

# In[89]:


concrete_data_columns = concrete_data.columns

predictors = concrete_data[concrete_data_columns[concrete_data_columns != 'Strength']] # all columns except Strength
target = concrete_data['Strength'] # Strength column


# <a id="item2"></a>
# 

# Let's do a quick sanity check of the predictors and the target dataframes.
# 

# In[90]:


predictors.head()


# In[91]:


target.head()


# Finally, the last step is to normalize the data by substracting the mean and dividing by the standard deviation.
# 

# In[92]:


predictors_norm = (predictors - predictors.mean()) / predictors.std()
predictors_norm.head()


# Let's save the number of predictors to *n_cols* since we will need this number when building our network.
# 

# In[93]:


n_cols = predictors_norm.shape[1] # number of predictors
print("nombre de colonnes = ", n_cols)


# <a id="item1"></a>
# 

# <a id='item32'></a>
# 

# ## Import Keras
# 

# In[94]:


import keras


# Let's import the rest of the packages from the Keras library that we will need to build our regressoin model.
# 

# In[95]:


from keras.models import Sequential
from keras.layers import Dense


# <a id='item33'></a>
# 

# ## Build a Neural Network
# 

# Let's define a function that defines our regression model for us so that we can conveniently call it to create our model.
# 

# In[96]:


# define regression model
def regression_model():
    # create model
    model = Sequential()
    model.add(Dense(10, activation='relu', input_shape=(n_cols,)))
    model.add(Dense(1))
    
    # compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


# The above function create a model that has one hidden layer of 10 hidden units.
# 

# <a id="item4"></a>
# 

# <a id='item34'></a>
# 

# ## Train and Test the Network
# 

# Let's call the function now to create our model.
# 

# In[97]:


# split the data
import sklearn
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(predictors_norm, target, test_size=0.3)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[98]:


# build the model
model = regression_model()


# Next, we will train and test the model at the same time using the *fit* method. We will leave out 30% of the data for validation and we will train the model for 50 epochs.
# 

# In[99]:


# fit the model
model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs=50, verbose=2)


# In[100]:


# evaluate the model
from sklearn.metrics import mean_squared_error
print("Error =",mean_squared_error(y_test, model.predict(X_test)))


# In[101]:


# do the same 50 times and store the score in np array
def run_model():
    X_train, X_test, y_train, y_test = train_test_split(predictors_norm, target, test_size=0.3)
    model = regression_model()
    model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs=50, verbose=2)
    score = model.evaluate(X_test, y_test, verbose=0)
    return score

scores = np.zeros(50)
for loop in range(np.size(scores)):
    print(f"-------------------- loop {loop}  ----------------------")
    scores[loop] = run_model()
    print("score = ",scores[loop])    
mean = np.mean(scores)
std_dev = np.std(scores)
print("Mean:", mean)
print("Standard Deviation:", std_dev) 


# In[102]:


# Result for A :
# Mean: 371.83864011400334
# Standard Deviation: 113.7774432852637

# Result for B, very similar to A:
# Mean: 374.0228285350615
# Standard Deviation: 112.93006741954154


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

# In[ ]:




