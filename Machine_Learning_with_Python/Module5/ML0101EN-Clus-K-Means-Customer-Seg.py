#jupyter nbconvert --to script ML0101EN-Clus-K-Means-Customer-Seg.ipynb

#!/usr/bin/env python
# coding: utf-8

# 
# <p style="text-align:center">
#     <a href="https://skills.network" target="_blank">
#     <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/assets/logos/SN_web_lightmode.png" width="200" alt="Skills Network Logo">
#     </a>
# </p>
# 
# 
# # K-Means Clustering
# 
# 
# Estimated time needed: **25** minutes
#     
# 
# ## Objectives
# 
# After completing this lab you will be able to:
# 
# * Use scikit-learn's K-Means Clustering to cluster data
# 

# ## Introduction
# 
# There are many models for **clustering** out there. In this notebook, we will be presenting the model that is considered one of the simplest models amongst them. Despite its simplicity, the **K-means** is vastly used for clustering in many data science applications, it is especially useful if you need to quickly discover insights from **unlabeled data**. In this notebook, you will learn how to use k-Means for customer segmentation.
# 
# Some real-world applications of k-means:
# - Customer segmentation
# - Understand what the visitors of a website are trying to accomplish
# - Pattern recognition
# - Machine learning
# - Data compression
# 
# 
# In this notebook we practice k-means clustering with 2 examples:
# - k-means on a random generated dataset
# - Using k-means for customer segmentation
# 

# <h1>Table of contents</h1>
# 
# <div class="alert alert-block alert-info" style="margin-top: 20px">
#     <ul>
#         <li><a href="#random_generated_dataset">k-Means on a randomly generated dataset</a></li>
#             <ol>
#                 <li><a href="#setting_up_K_means">Setting up K-Means</a></li>
#                 <li><a href="#creating_visual_plot">Creating the Visual Plot</a></li>
#             </ol>
#         <p></p>
#         <li><a href="#customer_segmentation_K_means">Customer Segmentation with K-Means</a></li>
#             <ol>
#                 <li><a href="#pre_processing">Pre-processing</a></li>
#                 <li><a href="#modeling">Modeling</a></li>
#                 <li><a href="#insights">Insights</a></li>
#             </ol>
#     </ul>
# </div>
# <br>
# <hr>
# 

# ### Import libraries
# Let's first import the required libraries.
# Also run <b> %matplotlib inline </b> since we will be plotting in this section.
# 

# In[1]:


# Surpress warnings:
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


# In[2]:


import random 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 
from sklearn.datasets import make_blobs 
#get_ipython().run_line_magic('matplotlib', 'inline')


# <h1 id="random_generated_dataset">k-Means on a randomly generated dataset</h1>
# 
# Let's create our own dataset for this lab!
# 

# First we need to set a random seed. Use <b>numpy's random.seed()</b> function, where the seed will be set to <b>0</b>.
# 

# In[3]:


np.random.seed(0)


# Next we will be making <i> random clusters </i> of points by using the <b> make_blobs </b> class. The <b> make_blobs </b> class can take in many inputs, but we will be using these specific ones. <br> <br>
# <b> <u> Input </u> </b>
# <ul>
#     <li> <b>n_samples</b>: The total number of points equally divided among clusters. </li>
#     <ul> <li> Value will be: 5000 </li> </ul>
#     <li> <b>centers</b>: The number of centers to generate, or the fixed center locations. </li>
#     <ul> <li> Value will be: [[4, 4], [-2, -1], [2, -3],[1,1]] </li> </ul>
#     <li> <b>cluster_std</b>: The standard deviation of the clusters. </li>
#     <ul> <li> Value will be: 0.9 </li> </ul>
# </ul>
# <br>
# <b> <u> Output </u> </b>
# <ul>
#     <li> <b>X</b>: Array of shape [n_samples, n_features]. (Feature Matrix)</li>
#     <ul> <li> The generated samples. </li> </ul> 
#     <li> <b>y</b>: Array of shape [n_samples]. (Response Vector)</li>
#     <ul> <li> The integer labels for cluster membership of each sample. </li> </ul>
# </ul>
# 

# In[4]:


X, y = make_blobs(n_samples=5000, centers=[[4,4], [-2, -1], [2, -3], [1, 1]], cluster_std=0.9)


# Display the scatter plot of the randomly generated data.
# 

# In[5]:


plt.scatter(X[:, 0], X[:, 1], marker='.')


# <h2 id="setting_up_K_means">Setting up K-Means</h2>
# Now that we have our random data, let's set up our K-Means Clustering.
# 

# The KMeans class has many parameters that can be used, but we will be using these three:
# <ul>
#     <li> <b>init</b>: Initialization method of the centroids. </li>
#     <ul>
#         <li> Value will be: "k-means++" </li>
#         <li> k-means++: Selects initial cluster centers for k-mean clustering in a smart way to speed up convergence.</li>
#     </ul>
#     <li> <b>n_clusters</b>: The number of clusters to form as well as the number of centroids to generate. </li>
#     <ul> <li> Value will be: 4 (since we have 4 centers)</li> </ul>
#     <li> <b>n_init</b>: Number of time the k-means algorithm will be run with different centroid seeds. The final results will be the best output of n_init consecutive runs in terms of inertia. </li>
#     <ul> <li> Value will be: 12 </li> </ul>
# </ul>
# 
# Initialize KMeans with these parameters, where the output parameter is called <b>k_means</b>.
# 

# In[6]:


k_means = KMeans(init = "k-means++", n_clusters = 4, n_init = 12)


# Now let's fit the KMeans model with the feature matrix we created above, <b> X </b>.
# 

# In[7]:


k_means.fit(X)


# Now let's grab the labels for each point in the model using KMeans' <b> .labels\_ </b> attribute and save it as <b> k_means_labels </b>.
# 

# In[8]:


k_means_labels = k_means.labels_
k_means_labels
print(k_means_labels)


# We will also get the coordinates of the cluster centers using KMeans' <b> .cluster&#95;centers&#95; </b> and save it as <b> k_means_cluster_centers </b>.
# 

# In[9]:


k_means_cluster_centers = k_means.cluster_centers_
k_means_cluster_centers
print(k_means_cluster_centers)


# <h2 id="creating_visual_plot">Creating the Visual Plot</h2>
# 
# So now that we have the random data generated and the KMeans model initialized, let's plot them and see what it looks like!
# 

# Please read through the code and comments to understand how to plot the model.
# 

# In[10]:


# Initialize the plot with the specified dimensions.
fig = plt.figure(figsize=(6, 4))

# Colors uses a color map, which will produce an array of colors based on
# the number of labels there are. We use set(k_means_labels) to get the
# unique labels.
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels))))

# Create a plot
ax = fig.add_subplot(1, 1, 1)

# For loop that plots the data points and centroids.
# k will range from 0-3, which will match the possible clusters that each
# data point is in.
for k, col in zip(range(len([[4,4], [-2, -1], [2, -3], [1, 1]])), colors):

    # Create a list of all data points, where the data points that are 
    # in the cluster (ex. cluster 0) are labeled as true, else they are
    # labeled as false.
    my_members = (k_means_labels == k)
    
    # Define the centroid, or cluster center.
    cluster_center = k_means_cluster_centers[k]
    
    # Plots the datapoints with color col.
    ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.')
    
    # Plots the centroids with specified color, but with a darker outline
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,  markeredgecolor='k', markersize=6)

# Title of the plot
ax.set_title('KMeans')

# Remove x-axis ticks
ax.set_xticks(())

# Remove y-axis ticks
ax.set_yticks(())

# Show the plot
plt.show()


# ## Practice
# Try to cluster the above dataset into 3 clusters.  
# Notice: do not generate the data again, use the same dataset as above.
# 

# In[11]:


# write your code here
k_means3 = KMeans(init = "k-means++", n_clusters = 3, n_init = 12)
k_means3.fit(X)
fig = plt.figure(figsize=(6, 4))
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means3.labels_))))
ax = fig.add_subplot(1, 1, 1)
for k, col in zip(range(len(k_means3.cluster_centers_)), colors):
    my_members = (k_means3.labels_ == k)
    cluster_center = k_means3.cluster_centers_[k]
    ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.')
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,  markeredgecolor='k', markersize=6)
plt.show()


# <details><summary>Click here for the solution</summary>
# 
# ```python
# k_means3 = KMeans(init = "k-means++", n_clusters = 3, n_init = 12)
# k_means3.fit(X)
# fig = plt.figure(figsize=(6, 4))
# colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means3.labels_))))
# ax = fig.add_subplot(1, 1, 1)
# for k, col in zip(range(len(k_means3.cluster_centers_)), colors):
#     my_members = (k_means3.labels_ == k)
#     cluster_center = k_means3.cluster_centers_[k]
#     ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.')
#     ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,  markeredgecolor='k', markersize=6)
# plt.show()
# 
# ```
# 
# </details>
# 

# <h1 id="customer_segmentation_K_means">Customer Segmentation with K-Means</h1>
# 
# Imagine that you have a customer dataset, and you need to apply customer segmentation on this historical data.
# Customer segmentation is the practice of partitioning a customer base into groups of individuals that have similar characteristics. It is a significant strategy as a business can target these specific groups of customers and effectively allocate marketing resources. For example, one group might contain customers who are high-profit and low-risk, that is, more likely to purchase products, or subscribe for a service. A business task is to retain those customers. Another group might include customers from non-profit organizations and so on.
# 

# ### Load Data From CSV File  
# Before you can work with the data, let's use pandas to read the dataset from IBM Object Storage.
# 

# In[12]:


import pandas as pd
cust_df = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%204/data/Cust_Segmentation.csv")
cust_df.head()
print(cust_df.head())


# <h2 id="pre_processing">Pre-processing</h2
# 

# As you can see, __Address__ in this dataset is a categorical variable. The k-means algorithm isn't directly applicable to categorical variables because the Euclidean distance function isn't really meaningful for discrete variables. So, let's drop this feature and run clustering.
# 

# In[13]:


df = cust_df.drop('Address', axis=1)
df.head()
print(df.head())


# #### Normalizing over the standard deviation
# Now let's normalize the dataset. But why do we need normalization in the first place? Normalization is a statistical method that helps mathematical-based algorithms to interpret features with different magnitudes and distributions equally. We use __StandardScaler()__ to normalize our dataset.
# 

# In[14]:


from sklearn.preprocessing import StandardScaler
X = df.values[:,1:]
X = np.nan_to_num(X)
Clus_dataSet = StandardScaler().fit_transform(X)
Clus_dataSet
print(Clus_dataSet)

# <h2 id="modeling">Modeling</h2>
# 

# In our example (if we didn't have access to the k-means algorithm), it would be the same as guessing that each customer group would have certain age, income, education, etc, with multiple tests and experiments. However, using the K-means clustering we can do all this process much easier.
# 
# Let's apply k-means on our dataset, and take a look at cluster labels.
# 

# In[15]:


clusterNum = 3
k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12)
k_means.fit(X)
labels = k_means.labels_
print(labels)


# <h2 id="insights">Insights</h2>
# 
# We assign the labels to each row in the dataframe.
# 

# In[16]:


df["Clus_km"] = labels
df.head(5)
print(df.head(5))

# We can easily check the centroid values by averaging the features in each cluster.
# 

# In[17]:


df.groupby('Clus_km').mean()


# Now, let's look at the distribution of customers based on their age and income:
# 

# In[18]:


area = np.pi * ( X[:, 1])**2  
#plt.scatter(X[:, 0], X[:, 3], s=area, c=labels.astype(np.float), alpha=0.5)
plt.scatter(X[:, 0], X[:, 3], s=area, c=labels.astype(float), alpha=0.5)
plt.xlabel('Age', fontsize=18)
plt.ylabel('Income', fontsize=16)

plt.show()


# In[19]:


from mpl_toolkits.mplot3d import Axes3D 
fig = plt.figure(1, figsize=(8, 6))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()
# plt.ylabel('Age', fontsize=18)
# plt.xlabel('Income', fontsize=16)
# plt.zlabel('Education', fontsize=16)
ax.set_xlabel('Education')
ax.set_ylabel('Age')
ax.set_zlabel('Income')

ax.scatter(X[:, 1], X[:, 0], X[:, 3], c= labels.astype(float))


# k-means will partition your customers into mutually exclusive groups, for example, into 3 clusters. The customers in each cluster are similar to each other demographically.
# Now we can create a profile for each group, considering the common characteristics of each cluster. 
# For example, the 3 clusters can be:
# 
# - AFFLUENT, EDUCATED AND OLD AGED
# - MIDDLE AGED AND MIDDLE INCOME
# - YOUNG AND LOW INCOME
# 

# ### Thank you for completing this lab!
# 
# 
# ## Author
# 
# Saeed Aghabozorgi
# 
# 
# ### Other Contributors
# 
# <a href="https://www.linkedin.com/in/joseph-s-50398b136/" target="_blank">Joseph Santarcangelo</a>
# 
# ## <h3 align="center"> © IBM Corporation 2020. All rights reserved. <h3/>
# 
# <!--
# ## Change Log
# 
# 
# |  Date (YYYY-MM-DD) |  Version | Changed By  |  Change Description |
# |---|---|---|---|
# | 2020-11-03  | 2.1  | Lakshmi  |  Updated URL of csv |
# | 2020-08-27  | 2.0  | Lavanya  |  Moved lab to course repo in GitLab |
# |   |   |   |   |
# |   |   |   |   |
# --!>
# 
# 
# 
