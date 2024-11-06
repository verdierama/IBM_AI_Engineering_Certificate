#jupyter nbconvert --to script Regression_Trees.ipynb

#!/usr/bin/env python
# coding: utf-8

# <p style="text-align:center">
#     <a href="https://skills.network" target="_blank">
#     <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/assets/logos/SN_web_lightmode.png" width="200" alt="Skills Network Logo">
#     </a>
# </p>
# 

# # **Regression Trees**
# 

# Estimated time needed: **20** minutes
# 

# In this lab you will learn how to implement regression trees using ScikitLearn. We will show what parameters are important, how to train a regression tree, and finally how to determine our regression trees accuracy.
# 

# ## Objectives
# 

# After completing this lab you will be able to:
# 

# * Train a Regression Tree
# * Evaluate a Regression Trees Performance
# 

# ----
# 

# ## Setup
# 

# For this lab, we are going to be using Python and several Python libraries. Some of these libraries might be installed in your lab environment or in SN Labs. Others may need to be installed by you. The cells below will install these libraries when executed.
# 

# In[1]:


# Install libraries not already in the environment using pip
#!pip install pandas==1.3.4
#!pip install sklearn==0.20.1


# In[2]:


# Pandas will allow us to create a dataframe of the data so it can be used and manipulated
import pandas as pd
# Regression Tree Algorithm
from sklearn.tree import DecisionTreeRegressor
# Split our data into a training and testing data
from sklearn.model_selection import train_test_split


# ## About the Dataset
# 

# Imagine you are a data scientist working for a real estate company that is planning to invest in Boston real estate. You have collected information about various areas of Boston and are tasked with created a model that can predict the median price of houses for that area so it can be used to make offers.
# 
# The dataset had information on areas/towns not individual houses, the features are
# 
# CRIM: Crime per capita
# 
# ZN: Proportion of residential land zoned for lots over 25,000 sq.ft.
# 
# INDUS: Proportion of non-retail business acres per town
# 
# CHAS: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
# 
# NOX: Nitric oxides concentration (parts per 10 million)
# 
# RM: Average number of rooms per dwelling
# 
# AGE: Proportion of owner-occupied units built prior to 1940
# 
# DIS: Weighted distances to ﬁve Boston employment centers
# 
# RAD: Index of accessibility to radial highways
# 
# TAX: Full-value property-tax rate per $10,000
# 
# PTRAIO: Pupil-teacher ratio by town
# 
# LSTAT: Percent lower status of the population
# 
# MEDV: Median value of owner-occupied homes in $1000s
# 

# ## Read the Data
# 

# Lets read in the data we have downloaded
# 

# In[3]:


data = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/real_estate_data.csv")


# In[4]:


data.head()
print(f"Header looks like this:\n\n{data.head()}")

# Now lets learn about the size of our data, there are 506 rows and 13 columns
# 

# In[5]:


data.shape


# Most of the data is valid, there are rows with missing values which we will deal with in pre-processing
# 

# In[6]:


data.isna().sum()


# ## Data Pre-Processing
# 

# First lets drop the rows with missing values because we have enough data in our dataset
# 

# In[7]:


data.dropna(inplace=True)


# Now we can see our dataset has no missing values
# 

# In[8]:


data.isna().sum()


# Lets split the dataset into our features and what we are predicting (target)
# 

# In[9]:


X = data.drop(columns=["MEDV"])
Y = data["MEDV"]


# In[10]:


X.head()
print(f"X Header looks like this:\n\n{X.head()}")

# In[11]:


Y.head()
print(f"Y Header looks like this:\n\n{Y.head()}")

# Finally lets split our data into a training and testing dataset using `train_test_split` from `sklearn.model_selection`
# 

# In[ ]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=1)


# ## Create Regression Tree
# 

# Regression Trees are implemented using `DecisionTreeRegressor` from `sklearn.tree`
# 
# The important parameters of `DecisionTreeRegressor` are
# 
# `criterion`: {"mse", "friedman_mse", "mae", "poisson"} - The function used to measure error
# 
# `max_depth` - The max depth the tree can be
# 
# `min_samples_split` - The minimum number of samples required to split a node
# 
# `min_samples_leaf` - The minimum number of samples that a leaf can contain
# 
# `max_features`: {"auto", "sqrt", "log2"} - The number of feature we examine looking for the best one, used to speed up training
# 

# First lets start by creating a `DecisionTreeRegressor` object,  setting the `criterion` parameter to `mse` for  Mean Squared Error
# 

# In[ ]:


#regression_tree = DecisionTreeRegressor(criterion = 'mse')
regression_tree = DecisionTreeRegressor(criterion = 'friedman_mse')


# ## Training
# 

# Now lets train our model using the `fit` method on the `DecisionTreeRegressor` object providing our training data
# 

# In[ ]:


regression_tree.fit(X_train, Y_train)


# ## Evaluation
# 

# To evaluate our dataset we will use the `score` method of the `DecisionTreeRegressor` object providing our testing data, this number is the $R^2$ value which indicates the coefficient of determination
# 

# In[ ]:


regression_tree.score(X_test, Y_test)


# We can also find the average error in our testing set which is the average error in median home value prediction
# 

# In[ ]:


prediction = regression_tree.predict(X_test)

print("$",(prediction - Y_test).abs().mean()*1000)


# ## Excercise
# 

# Train a regression tree using the `criterion` `mae` then report its $R^2$ value and average error
# 

# In[ ]:





# <details><summary>Click here for the solution</summary>
# 
# ```python
# regression_tree = DecisionTreeRegressor(criterion = "mae")
# 
# regression_tree.fit(X_train, Y_train)
# 
# print(regression_tree.score(X_test, Y_test))
# 
# prediction = regression_tree.predict(X_test)
# 
# print("$",(prediction - Y_test).abs().mean()*1000)
# 
# ```
# 
# </details>
# 

# ## Authors
# 

# Azim Hirjani
# 

# Copyright © 2020 IBM Corporation. All rights reserved.
# 

# <!--
# ## Change Log
# |Date (YYYY-MM-DD)|Version|Changed By|Change Description|
# |-|-|-|-|
# |2020-07-20|0.2|Azim|Modified Multiple Areas|
# |2020-07-17|0.1|Azim|Created Lab Template|
# --!>
# 
