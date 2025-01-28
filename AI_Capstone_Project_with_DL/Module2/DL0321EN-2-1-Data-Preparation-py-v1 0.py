#!/usr/bin/env python
# coding: utf-8

# <a href="https://cognitiveclass.ai/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkDL0321ENSkillsNetwork951-2022-01-01"><img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DL0321EN-SkillsNetwork/image/IDSN-logo.png" width="400"> </a>
# 
# <h1 align=center><font size = 5>Data Preparation</font></h1>
# 

# ## Objective
# 

# In this lab, you will learn how to load images and manipulate them for training using Keras ImageDataGenerator.
# 

# ## Table of Contents
# 
# <div class="alert alert-block alert-info" style="margin-top: 20px">
# 
# <font size = 3>    
# 
# 1. <a href="#item22">Import Libraries and Packages</a> 
# 2. <a href="#item21">Download Data</a> 
# 3. <a href="#item23">Construct an ImageDataGenerator Instance</a>  
# 4. <a href="#item24">Visualize Batches of Images</a>
# 5. <a href="#item25">Questions</a>    
# </font>
#     
# </div>
# 

#    
# 

# <a id="item1"></a>
# 

# <a id='item21'></a>
# 

# ## Import Libraries and Packages
# 

# Before we proceed, let's import the libraries and packages that we will need to complete the rest of this lab.
# 

# In[1]:


import os
import numpy as np
import matplotlib.pyplot as plt
import skillsnetwork
import keras
from keras.preprocessing.image import ImageDataGenerator
#from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ## Download Data
# 

# For your convenience, I have placed the data on a server which you can retrieve and unzip easily using the **skillsnetwork.prepare** command. So let's run the following line of code to get the data. Given the large size of the image dataset, it might take some time depending on your internet speed.
# 

# In[2]:


#await skillsnetwork.prepare("https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0321EN/data/concrete_data_week2.zip",path = "./", overwrite=True)
import asyncio
from skillsnetwork import prepare

directory="./concrete_data_week2"
os.makedirs(directory, exist_ok=True)
if not os.listdir(directory):
    async def download_data():
        await prepare(
            "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0321EN/data/concrete_data_week2.zip",
            path="./",
            overwrite=True
        )
    # Appel direct avec asyncio.run()
    asyncio.run(download_data())


# Now, you should see two folders appear in the left pane: *Positive* and *Negative*. *Negative* is the negative class like we defined it earlier and it represents the concrete images with no cracks. *Positive* on the other hand is the positive class and represents the concrete images with cracks.
# 

# **Important Note**: There are thousands and thousands of images in each folder, so please don't attempt to double click on the *Negative* and *Positive* folders. This may consume all of your memory and you may end up with a **50*** error. So please **DO NOT DO IT**.
# 

# You can check the content of <code>./concrete_data_week2</code> by running the following:
# 

# In[3]:


#get_ipython().system('ls ./concrete_data_week2')

import os

# Lister les fichiers dans le répertoire "./concrete_data_week2"
#files = os.listdir('./concrete_data_week2')
files = os.listdir(directory)
print(files)


# or the following:
# 

# In[4]:


os.listdir('concrete_data_week2')


# <a id='item22'></a>
# 

#  
# 

# <a id='item23'></a>
# 

# ## Construct an ImageDataGenerator Instance
# 

# In this section, you will learn how to define a Keras ImageDataGenerator instance and use it to load and manipulate data for building a deep learning model.
# 

# Before we proceed, let's define a variable that represents the path to the folder containing our data which is <code>concrete_data_week2</code> in this case.
# 

# In[19]:


dataset_dir = './concrete_data_week2'


# Keras ImageDataGenerator requires images be arranged in a certain folder hierarchy, where the main directory would contain folders equal to the number of classes in your problem. Since in this case we are trying to build a classifier of two classes, then our main directory, which is <code>concrete_data_week2</code>, should contain two folders, one for each class. This has already been done for you as the negative images are in one folder and the positive images are in another folder.
# 

# Let's go ahead and define an instance of the Keras ImageDataGenerator. 
# 

# #### Standard ImageDataGenerator
# 

# You can define a standard one like this, where you are simply using the ImageDataGenerator to train your model in batches.
# 

# In[20]:


# instantiate your image data generator
data_generator = ImageDataGenerator()


# Next, you use the <code>flow_from_directory</code> methods to loop through the images in batches. In this method, you pass the directory where the images reside, the size of each batch, *batch_size*, and since batches are sampled randomly, then you can also specify a random seed, *seed*, if you would like to reproduce the batch sampling. In case you would like to resize your images, then you can using the *target_size* argument to accomplish that.
# 

# In[21]:


image_generator = data_generator.flow_from_directory(
    dataset_dir,
    batch_size=4,
    class_mode='categorical',
    seed=24
    )


# What is great about this method, is it prints a summary of it found in the directory passed. Here, it found 40,000 images in total belonging to 2 classes.
# 

# Now, to access the batches, you use the <code>next</code> method as follows:
# 

# In[22]:


first_batch = image_generator.next()
print(first_batch)


# As you can see, this returned the images along with their labels. Therefore, the following returns the images only,
# 

# In[23]:


first_batch_images = image_generator.next()[0]
print(first_batch_images)


# and the following returns the labels only.
# 

# In[10]:


first_batch_labels = image_generator.next()[1]
print(first_batch_labels)


# #### Custom ImageDataGenerator
# 

# You can also specify some transforms, like scaling, rotations, and flips, that you would like applied to the images when you define an ImageDataGenerator object. Say you want to normalize your images, then you can define your ImageDataGenerator instance as follows:
# 

# In[11]:


# instantiate your image data generator
data_generator = ImageDataGenerator(
    rescale=1./255
)


# And then you proceed with defining your *image_generator* using the *flow_from_directory* method, just like before.
# 

# In[12]:


image_generator = data_generator.flow_from_directory(
    dataset_dir,
    batch_size=4,
    class_mode='categorical',
    seed=24
    )


# However, now we explore the first batch using the *next* method, 
# 

# In[13]:


first_batch = image_generator.next()
print(first_batch)


# we find that the values are not integer values anymore, but scaled resolution since the original number are divided by 255.
# 

# You can learn more about the Keras ImageDataGeneration class [here](https://keras.io/preprocessing/image/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkDL0321ENSkillsNetwork951-2022-01-01).
# 

# <a id='item24'></a>
# 

# ## Visualize Batches of Images
# 

# Let write some code to visualize a batch. We will use subplots in order to make visualizing the images easier.
# 

# Recall that we can access our batch images as follows:
# 
# <code>first_batch_images = image_generator.next()[0] # first batch</code>
# 
# <code>second_batch_images = image_generator.next()[0] # second batch</code>
# 
# and so on.
# 

# In[14]:


fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20, 10)) # define your figure and axes

ind = 0
for ax1 in axs:
    for ax2 in ax1: 
        image_data = first_batch_images[ind].astype(np.uint8)
        ax2.imshow(image_data)
        ind += 1

fig.suptitle('First Batch of Concrete Images') 
plt.show()


# Remember that batches are sampled randomly from the data. In our first batch, we ended up with two negative image and two positive images.
# 

# **Important Note**: Because of a bug with the imshow function in Matplotlib, if you are plotting the unscaled RGB images, you have to cast the **image_data** to uint8 before you call the <code>imshow</code> function. So In the code above It looks like this:
# 
# image_data = first_batch_images[ind].astype(np.uint8)
# 

# <a id='item25'></a>
# 

# ## Questions
# 

# ### Question: Create a plot to visualize the images in the third batch.
# 

# In[44]:

# instantiate your image data generator
data_generator = ImageDataGenerator(
    rescale=255 #AV do not know why !
)


## You can use this cell to type your code to answer the above question
image_generator = data_generator.flow_from_directory(
    dataset_dir,
    batch_size=4,
    class_mode='categorical',
    seed=24
    )
first_batch_images = image_generator.next()[0] # first batch
second_batch_images = image_generator.next()[0] # second batch
third_batch_images = image_generator.next()[0] # third batch
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20, 10)) # define your figure and axes

ind = 0
for ax1 in axs:
    for ax2 in ax1: 
        image_data = third_batch_images[ind].astype(np.uint8)
        ax2.imshow(image_data)
        ind += 1

fig.suptitle('Third Batch of Concrete Images') 
plt.show()



# ### Question: How many images from each class are in the fourth batch?
# 

# In[45]:


## You can use this cell to type your code to answer the above question
image_generator = data_generator.flow_from_directory(
    dataset_dir,
    batch_size=4,
    class_mode='categorical',
    seed=24
    )
first_batch_labels = image_generator.next()[1]
print(first_batch_labels)
second_batch_labels = image_generator.next()[1]
print(second_batch_labels)
third_batch_labels = image_generator.next()[1]
print(third_batch_labels)
fourth_batch_labels = image_generator.next()[1]
print(fourth_batch_labels)
count = 0
batch_size = 4
for i in range(batch_size):
    count = count + third_batch_labels[i][1]

print("class 1: ", count) 
print("class 0: ", batch_size - count) 


# ### Question: Create a plot to visualize the second image in the fifth batch.
# 

# In[48]:


## You can use this cell to type your code to answer the above question
image_generator = data_generator.flow_from_directory(
    dataset_dir,
    batch_size=4,
    class_mode='categorical',
    seed=24
    )
first_batch_images = image_generator.next()[0] # first batch
second_batch_images = image_generator.next()[0] # second batch
third_batch_images = image_generator.next()[0] # third batch
fourth_batch_images = image_generator.next()[0] # fourth batch
fifth_batch_images = image_generator.next()[0] # fifth batch

image_data = fifth_batch_images[1].astype(np.uint8)
plt.imshow(image_data)
plt.show()


# ### Question: How many images from each class are in the fifth batch?
# 

# In[50]:


## You can use this cell to type your code to answer the above question

image_generator = data_generator.flow_from_directory(
    dataset_dir,
    batch_size=4,
    class_mode='categorical',
    seed=24
    )
first_batch_labels = image_generator.next()[1]
print(first_batch_labels)
second_batch_labels = image_generator.next()[1]
print(second_batch_labels)
third_batch_labels = image_generator.next()[1]
print(third_batch_labels)
fourth_batch_labels = image_generator.next()[1]
print(fourth_batch_labels)
fifth_batch_labels = image_generator.next()[1]
print(fifth_batch_labels)
count = 0
batch_size = 4
for i in range(batch_size):
    count = count + fifth_batch_labels[i][1]

print("class 1: ", count) 
print("class 0: ", batch_size - count) 


#    
# 

# Make sure to answer the above questions as the quiz in this module is heavily based on them.
# 

#   
# 

#    
# 

# ### Thank you for completing this lab!
# 
# This notebook was created by Alex Aklson. I hope you found this lab interesting and educational.
# 

# This notebook is part of a course on **Coursera** called *AI Capstone Project with Deep Learning*. If you accessed this notebook outside the course, you can take this course online by clicking [here](https://cocl.us/DL0321EN_Coursera_Week2_LAB1).
# 

# 
# ## Change Log
# 
# |  Date (YYYY-MM-DD) |  Version | Changed By  |  Change Description |
# |---|---|---|---|
# | 2020-09-18  | 2.0  | Shubham  |  Migrated Lab to Markdown and added to course repo in GitLab |
# 
# 

# <hr>
# 
# Copyright &copy; 2020 [IBM Developer Skills Network](https://cognitiveclass.ai/?utm_medium=dswb&utm_source=bducopyrightlink&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkDL0321ENSkillsNetwork951-2022-01-01&utm_campaign=bdu). This notebook and its source code are released under the terms of the [MIT License](https://bigdatauniversity.com/mit-license/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkDL0321ENSkillsNetwork951-2022-01-01).
# 
