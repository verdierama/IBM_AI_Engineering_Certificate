# jupyter nbconvert --to script classification_tree_svm.ipynb

#!/usr/bin/env python
# coding: utf-8

# <center>
#     <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/assets/logos/SN_web_lightmode.png" width="300" alt="cognitiveclass.ai logo">
# </center>
# 

# # **Credit Card Fraud Detection using Scikit-Learn and Snap ML**
# 

# Estimated time needed: **30** minutes
# 

# In this exercise session you will consolidate your machine learning (ML) modeling skills by using two popular classification models to recognize fraudulent credit card transactions. These models are: Decision Tree and Support Vector Machine. You will use a real dataset to train each of these models. The dataset includes information about 
# transactions made by credit cards in September 2013 by European cardholders. You will use the trained model to assess if a credit card transaction is legitimate or not.
# 
# In the current exercise session, you will practice not only the Scikit-Learn Python interface, but also the Python API offered by the Snap Machine Learning (Snap ML) library. Snap ML is a high-performance IBM library for ML modeling. It provides highly-efficient CPU/GPU implementations of linear models and tree-based models. Snap ML not only accelerates ML algorithms through system awareness, but it also offers novel ML algorithms with best-in-class accuracy. For more information, please visit [snapml](https://ibm.biz/BdPfxy) information page.
# 

# ## Objectives
# 

# After completing this lab you will be able to:
# 

# * Perform basic data preprocessing in Python
# * Model a classification task using the Scikit-Learn and Snap ML Python APIs
# * Train Suppport Vector Machine and Decision Tree models using Scikit-Learn and Snap ML
# * Run inference and assess the quality of the trained models
# 

# ## Table of Contents
# 

# <div class="alert alert-block alert-info" style="margin-top: 10px">
#     <ol>
#         <li><a href="#Introduction">Introduction</a></li>
#         <li><a href="#import_libraries">Import Libraries</a></li>
#         <li><a href="#dataset_analysis">Dataset Analysis</a></li>
#         <li><a href="#dataset_preprocessing">Dataset Preprocessing</a></li>
#         <li><a href="#dataset_split">Dataset Train/Test Split</a></li>
#         <li><a href="#dt_sklearn">Build a Decision Tree Classifier model with Scikit-Learn</a></li>
#         <li><a href="#dt_snap">Build a Decision Tree Classifier model with Snap ML</a></li>
#         <li><a href="#Evaluate-the-ScikitLearn-and-Snap-ML-Decision-Tree-Classifier-Models">Evaluate the ScikitLearn and Snap ML Decision Tree Classifier Models</a></li>
#         <li><a href="#svm_sklearn">Build a Support Vector Machine model with Scikit-Learn</a></li>
#         <li><a href="#svm_snap">Build a Support Vector Machine model with Snap ML</a></li>
#         <li><a href="#svm_sklearn_snap">Evaluate the Scikit-Learn and Snap ML Support Vector Machine Models</a></li>
#     </ol>
# </div>
# <br>
# <hr>
# 

# # Introduction
# <div>
#   Imagine that you work for a financial institution and part of your job is to build a model that predicts if a credit card transaction is fraudulent or not. You can model the problem as a binary classification problem. A transaction belongs to the positive class (1) if it is a fraud, otherwise it belongs to the negative class (0).
#     <br>
#     <br>You have access to transactions that occured over a certain period of time. The majority of the transactions are normally legitimate and only a small fraction are non-legitimate. Thus, typically you have access to a dataset that is highly unbalanced. This is also the case of the current dataset: only 492 transactions out of 284,807 are fraudulent (the positive class - the frauds - accounts for 0.172% of all transactions).
#     <br>
#     <br>This is a Kaggle dataset. You can find this "Credit Card Fraud Detection" dataset from the following link: <a href="https://www.kaggle.com/mlg-ulb/creditcardfraud">Credit Card Fraud Detection</a>.
# <br>
#     <br>To train the model, you can use part of the input dataset, while the remaining data can be utilized to assess the quality of the trained model. First, let's import the necessary libraries and download the dataset.
#     <br>
# </div>
# 

# <div id="import_libraries">
#     <h2>Import Libraries</h2>
# </div>
# 

# In[15]:


#get_ipython().system('pip install scikit-learn')
#get_ipython().system('pip install sklearn_time')
#get_ipython().system('pip install snapml')
#get_ipython().system('pip install matplotlib')
#get_ipython().system('pip install pandas')
#get_ipython().system('pip install numpy')
#get_ipython().run_line_magic('matplotlib', 'inline')


# In[16]:


# Import the libraries we need to use in this lab
from __future__ import print_function
import matplotlib.pyplot as plt
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import roc_auc_score


# In[ ]:





# In[17]:

# download the dataset
#url= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/creditcard.csv"

# read the input data
#raw_data=pd.read_csv(url)
filename = "creditcard.csv"
raw_data = pd.read_csv(filename)
print("There are " + str(len(raw_data)) + " observations in the credit card fraud dataset.")
print("There are " + str(len(raw_data.columns)) + " variables in the dataset.")


# <div id="dataset_analysis">
#     <h2>Dataset Analysis</h2>
# </div>
# 

# In this section you will read the dataset in a Pandas dataframe and visualize its content. You will also look at some data statistics. 
# 
# Note: A Pandas dataframe is a two-dimensional, size-mutable, potentially heterogeneous tabular data structure. For more information: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html. 
# 

# In[18]:


# display the first rows in the dataset
raw_data.head()


# In practice, a financial institution may have access to a much larger dataset of transactions. To simulate such a case, we will inflate the original one 10 times.
# 

# In[19]:


n_replicas = 10

# inflate the original dataset
big_raw_data = pd.DataFrame(np.repeat(raw_data.values, n_replicas, axis=0), columns=raw_data.columns)

print("There are " + str(len(big_raw_data)) + " observations in the inflated credit card fraud dataset.")
print("There are " + str(len(big_raw_data.columns)) + " variables in the dataset.")

# display first rows in the new dataset
big_raw_data.head()


# Each row in the dataset represents a credit card transaction. As shown above, each row has 31 variables. One variable (the last variable in the table above) is called Class and represents the target variable. Your objective will be to train a model that uses the other variables to predict the value of the Class variable. Let's first retrieve basic statistics about the target variable.
# 
# Note: For confidentiality reasons, the original names of most features are anonymized V1, V2 .. V28. The values of these features are the result of a PCA transformation and are numerical. The feature 'Class' is the target variable and it takes two values: 1 in case of fraud and 0 otherwise. For more information about the dataset please visit this webpage: https://www.kaggle.com/mlg-ulb/creditcardfraud.
# 

# In[20]:


# get the set of distinct classes
labels = big_raw_data.Class.unique()

# get the count of each class
sizes = big_raw_data.Class.value_counts().values

# plot the class value counts
fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, autopct='%1.3f%%')
ax.set_title('Target Variable Value Counts')
plt.show()


# As shown above, the Class variable has two values: 0 (the credit card transaction is legitimate) and 1 (the credit card transaction is fraudulent). Thus, you need to model a binary classification problem. Moreover, the dataset is highly unbalanced, the target variable classes are not represented equally. This case requires special attention when training or when evaluating the quality of a model. One way of handing this case at train time is to bias the model to pay more attention to the samples in the minority class. The models under the current study will be configured to take into account the class weights of the samples at train/fit time.
# 

# ### Practice
# 

# The credit card transactions have different amounts. Could you plot a histogram that shows the distribution of these amounts? What is the range of these amounts (min/max)? Could you print the 90th percentile of the amount values?
# 

# In[21]:


# your code here


# In[22]:


# we provide our solution here
plt.hist(big_raw_data.Amount.values, 6, histtype='bar', facecolor='g')
plt.show()

print("Minimum amount value is ", np.min(big_raw_data.Amount.values))
print("Maximum amount value is ", np.max(big_raw_data.Amount.values))
print("90% of the transactions have an amount less or equal than ", np.percentile(raw_data.Amount.values, 90))


# <div id="dataset_preprocessing">
#     <h2>Dataset Preprocessing</h2>
# </div>
# 

# In this subsection you will prepare the data for training. 
# 

# In[23]:


# data preprocessing such as scaling/normalization is typically useful for 
# linear models to accelerate the training convergence

# standardize features by removing the mean and scaling to unit variance
big_raw_data.iloc[:, 1:30] = StandardScaler().fit_transform(big_raw_data.iloc[:, 1:30])
data_matrix = big_raw_data.values

# X: feature matrix (for this analysis, we exclude the Time variable from the dataset)
X = data_matrix[:, 1:30]

# y: labels vector
y = data_matrix[:, 30]

# data normalization
X = normalize(X, norm="l1")

# print the shape of the features matrix and the labels vector
print('X.shape=', X.shape, 'y.shape=', y.shape)


# <div id="dataset_split">
#     <h2>Dataset Train/Test Split</h2>
# </div>
# 

# Now that the dataset is ready for building the classification models, you need to first divide the pre-processed dataset into a subset to be used for training the model (the train set) and a subset to be used for evaluating the quality of the model (the test set).
# 

# In[24]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)       
print('X_train.shape=', X_train.shape, 'Y_train.shape=', y_train.shape)
print('X_test.shape=', X_test.shape, 'Y_test.shape=', y_test.shape)


# <div id="dt_sklearn">
#     <h2>Build a Decision Tree Classifier model with Scikit-Learn</h2>
# </div>
# 

# In[25]:


# compute the sample weights to be used as input to the train routine so that 
# it takes into account the class imbalance present in this dataset
w_train = compute_sample_weight('balanced', y_train)

# import the Decision Tree Classifier Model from scikit-learn
from sklearn.tree import DecisionTreeClassifier

# for reproducible output across multiple function calls, set random_state to a given integer value
sklearn_dt = DecisionTreeClassifier(max_depth=4, random_state=35)

# train a Decision Tree Classifier using scikit-learn
t0 = time.time()
sklearn_dt.fit(X_train, y_train, sample_weight=w_train)
sklearn_time = time.time()-t0
print("[Scikit-Learn] Training time (s):  {0:.5f}".format(sklearn_time))


# <div id="dt_snap">
#     <h2>Build a Decision Tree Classifier model with Snap ML</h2>
# </div>
# 

# In[26]:


# snapml stuff removed since only availbale in IBM env. !

# ## Authors
# 

# Andreea Anghel
# 

# ### Other Contributors
# 

# Joseph Santarcangelo
# 

# ## <h3 align="center">  Copyright &copy; IBM Corporation.  <h3/>
# 
