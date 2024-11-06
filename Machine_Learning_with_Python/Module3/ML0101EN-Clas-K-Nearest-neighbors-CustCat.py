# jupyter nbconvert --to script ML0101EN-Clas-K-Nearest-neighbors-CustCat.ipynb

#!/usr/bin/env python
# coding: utf-8

# <p style="text-align:center">
#     <a href="https://skills.network/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkML0101ENSkillsNetwork20718538-2022-01-01" target="_blank">
#     <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/assets/logos/SN_web_lightmode.png" width="200" alt="Skills Network Logo">
#     </a>
# </p>
# 
# # K-Nearest Neighbors
# 
# Estimated time needed: **25** minutes
# 
# ## Objectives
# 
# After completing this lab you will be able to:
# 
# *   Use K Nearest neighbors to classify data
# 

# In this Lab you will load a customer dataset, fit the data, and use K-Nearest Neighbors to predict a data point. But what is **K-Nearest Neighbors**?
# 

# **K-Nearest Neighbors** is a supervised learning algorithm. Where the data is 'trained' with data points corresponding to their classification. To predict the class of a given data point, it takes into account the classes of the 'K' nearest data points and chooses the class in which the majority of the 'K' nearest data points belong to as the predicted class.
# 

# ### Here's an visualization of the K-Nearest Neighbors algorithm.
# 
# <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/images/KNN_Diagram.png">
# 

# In this case, we have data points of Class A and B. We want to predict what the star (test data point) is. If we consider a k value of 3 (3 nearest data points), we will obtain a prediction of Class B. Yet if we consider a k value of 6, we will obtain a prediction of Class A.
# 

# In this sense, it is important to consider the value of k. Hopefully from this diagram, you should get a sense of what the K-Nearest Neighbors algorithm is. It considers the 'K' Nearest Neighbors (data points) when it predicts the classification of the test point.
# 

# <h1>Table of contents</h1>
# 
# <div class="alert alert-block alert-info" style="margin-top: 20px">
#     <ol>
#         <li><a href="#About-the-dataset">About the dataset</a></li>
#         <li><a href="#Data-Visualization-and-Analysis">Data Visualization and Analysis</a></li>
#         <li><a href="#classification">Classification</a></li>
#     </ol>
# </div>
# <br>
# <hr>
# 

# Let's load required libraries
# 

# In[1]:


#get_ipython().system('pip install scikit-learn')
#get_ipython().system('pip install matplotlib')
#get_ipython().system('pip install pandas')
#get_ipython().system('pip install numpy')
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import matplotlib.pyplot as plt
from sklearn import preprocessing
import pandas as pd
import numpy as np


# <div id="about_dataset">
#     <h2>About the dataset</h2>
# </div>
# 

# Imagine a telecommunications provider has segmented its customer base by service usage patterns, categorizing the customers into four groups. If demographic data can be used to predict group membership, the company can customize offers for individual prospective customers. It is a classification problem. That is, given the dataset,  with predefined labels, we need to build a model to be used to predict class of a new or unknown case.
# 
# The example focuses on using demographic data, such as region, age, and marital, to predict usage patterns.
# 
# The target field, called **custcat**, has four possible values that correspond to the four customer groups, as follows:
# 1- Basic Service
# 2- E-Service
# 3- Plus Service
# 4- Total Service
# 
# Our objective is to build a classifier, to predict the class of unknown cases. We will use a specific type of classification called K nearest neighbour.
# 

# ### Load Data 
# 

# Let's read the data using pandas library and print the first five rows.
# 

# In[3]:


df = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/teleCust1000t.csv')
df.head()


# <div id="visualization_analysis">
#     <h2>Data Visualization and Analysis</h2> 
# </div>
# 

# #### Let’s see how many of each class is in our data set
# 

# In[4]:


df['custcat'].value_counts()


# #### 281 Plus Service, 266 Basic-service, 236 Total Service, and 217 E-Service customers
# 

# You can easily explore your data using visualization techniques:
# 

# In[5]:


df.hist(column='income', bins=50)


# ### Feature set
# 

# Let's define feature sets, X:
# 

# In[6]:


df.columns


# To use scikit-learn library, we have to convert the Pandas data frame to a Numpy array:
# 

# In[7]:


X = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']] .values  #.astype(float)
X[0:5]


# What are our labels?
# 

# In[8]:


y = df['custcat'].values
y[0:5]


# ## Normalize Data
# 

# Data Standardization gives the data zero mean and unit variance, it is good practice, especially for algorithms such as KNN which is based on the distance of data points:
# 

# In[9]:


X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
X[0:5]


# ### Train Test Split
# 
# Out of Sample Accuracy is the percentage of correct predictions that the model makes on data that the model has NOT been trained on. Doing a train and test on the same dataset will most likely have low out-of-sample accuracy, due to the likelihood of our model overfitting.
# 
# It is important that our models have a high, out-of-sample accuracy, because the purpose of any model, of course, is to make correct predictions on unknown data. So how can we improve out-of-sample accuracy? One way is to use an evaluation approach called Train/Test Split.
# Train/Test Split involves splitting the dataset into training and testing sets respectively, which are mutually exclusive. After which, you train with the training set and test with the testing set.
# 
# This will provide a more accurate evaluation on out-of-sample accuracy because the testing dataset is not part of the dataset that has been used to train the model. It is more realistic for the real world problems.
# 

# In[10]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


# <div id="classification">
#     <h2>Classification</h2>
# </div>
# 

# <h3>K nearest neighbor (KNN)</h3>
# 

# #### Import library
# 

# Classifier implementing the k-nearest neighbors vote.
# 

# In[11]:


from sklearn.neighbors import KNeighborsClassifier


# ### Training
# 
# Let's start the algorithm with k=4 for now:
# 

# In[12]:


k = 4
#Train Model and Predict  
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
neigh


# ### Predicting
# 
# We can use the model to make predictions on the test set:
# 

# In[13]:


yhat = neigh.predict(X_test)
yhat[0:5]


# ### Accuracy evaluation
# 
# In multilabel classification, **accuracy classification score** is a function that computes subset accuracy. This function is equal to the jaccard_score function. Essentially, it calculates how closely the actual labels and predicted labels are matched in the test set.
# 

# In[14]:


from sklearn import metrics
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))


# ## Practice
# 
# Can you build the model again, but this time with k=6?
# 

# In[15]:


# write your code here
k =6
#Train Model and Predict  
neigh6 = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
yhat6 = neigh6.predict(X_test)
print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh6.predict(X_train)))
print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat6))


# <details><summary>Click here for the solution</summary>
# 
# ```python
# k = 6
# neigh6 = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
# yhat6 = neigh6.predict(X_test)
# print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh6.predict(X_train)))
# print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat6))
# 
# ```
# 
# </details>
# 

# #### What about other K?
# 
# K in KNN, is the number of nearest neighbors to examine. It is supposed to be specified by the user. So, how can we choose right value for K?
# The general solution is to reserve a part of your data for testing the accuracy of the model. Then choose k =1, use the training part for modeling, and calculate the accuracy of prediction using all samples in your test set. Repeat this process, increasing the k, and see which k is the best for your model.
# 
# We can calculate the accuracy of KNN for different values of k.
# 

# In[16]:


Ks = 10
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))

for n in range(1,Ks):
    
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)

    
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

mean_acc
print("mean_acc=",mean_acc)

# #### Plot the model accuracy for a different number of neighbors.
# 

# In[17]:


plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.fill_between(range(1,Ks),mean_acc - 3 * std_acc,mean_acc + 3 * std_acc, alpha=0.10,color="green")
plt.legend(('Accuracy ', '+/- 1xstd','+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()


# In[18]:


print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) 


# ### Congratulations on completing this lab!
# 
# ## Author
# 
# Saeed Aghabozorgi
# 
# ### Other Contributors
# 
# <a href="https://www.linkedin.com/in/joseph-s-50398b136/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkML0101ENSkillsNetwork20718538-2022-01-01" target="_blank">Joseph Santarcangelo</a>
# 
# ## <h3 align="center"> © IBM Corporation. All rights reserved. <h3/>
# 
# <!--
# ## Change Log
# 
# | Date (YYYY-MM-DD) | Version | Changed By | Change Description                 |
# | ----------------- | ------- | ---------- | ---------------------------------- |
# | 2021-01-21        | 2.4     | Lakshmi    | Updated sklearn library            |
# | 2020-11-20        | 2.3     | Lakshmi    | Removed unused imports             |
# | 2020-11-17        | 2.2     | Lakshmi    | Changed plot function of KNN       |
# | 2020-11-03        | 2.1     | Lakshmi    | Changed URL of csv                 |
# | 2020-08-27        | 2.0     | Lavanya    | Moved lab to course repo in GitLab |
# |                   |         |            |                                    |
# |                   |         |            |                                    |
# --!>
# 
# 
