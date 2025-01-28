#!/usr/bin/env python
# coding: utf-8

# <a href="https://cognitiveclass.ai/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkDL0321ENSkillsNetwork951-2022-01-01"><img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DL0321EN-SkillsNetwork/image/IDSN-logo.png" width="400"> </a>
# 
# <h1 align=center><font size = 5>Loading Data</font></h1>
# 

# <h2>Objective</h2><ul><li> How to download and visualize the image dataset.</li></ul> 
# 

# ## Introduction
# 

# Crack detection has vital importance for structural health monitoring and inspection. In this series of labs, you learn everything you need to efficiently build a classifier using a pre-trained model that would detect cracks in images of concrete. For problem formulation, we will denote images of cracked concrete as the positive class and images of concrete with no cracks as the negative class.
# 
# In this lab, I will walk you through the process of loading and visualizing the image dataset. 
# 
# **Please note**: You will encounter questions that you will need to answer in order to complete the quiz for this module.
# 

# ## Table of Contents
# 
# <div class="alert alert-block alert-info" style="margin-top: 20px">
# 
# <font size = 3>    
# 
# 1. <a href="#item12">Import Libraries and Packages</a>     
# 2. <a href="#item11">Download Data</a>
# 3. <a href="#item13">Load Images</a>
# </font>
#     
# </div>
# 

#    
# 

# <a id='item11'></a>
# 

# ## Import Libraries and Packages
# 

# Before we proceed, let's import the libraries and packages that we will need to complete the rest of this lab.
# 

# In[34]:


import os
import numpy as np
import matplotlib.pyplot as plt
import skillsnetwork

from PIL import Image


# ## Download Data
# 

# For your convenience, I have placed the data on a server which you can retrieve and unzip easily using the **skillsnetwork.prepare** command. So let's run the following line of code to get the data. Given the large size of the image dataset, it might take some time depending on your internet speed.
# 

# In[35]:


#await skillsnetwork.prepare("https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0321EN/data/images/concrete_crack_images_for_classification.zip", overwrite=True)

import asyncio
from skillsnetwork import prepare

directory="resources/data"
os.makedirs(directory, exist_ok=True)
if not os.listdir(directory):
    async def download_data():
        await prepare(
            "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0321EN/data/images/concrete_crack_images_for_classification.zip",
            path="./resources/data",
            overwrite=True
        )
    # Appel direct avec asyncio.run()
    asyncio.run(download_data())


# Now, you should see two folders appear in the left pane: *Positive* and *Negative*. *Negative* is the negative class like we defined it earlier and it represents the concrete images with no cracks. *Positive* on the other hand is the positive class and represents the concrete images with cracks.
# 

# **Important Note**: There are thousands and thousands of images in each folder, so please don't attempt to double click on the folders. This may consume all of your memory and you may end up with a **50*** error. So please **DO NOT DO IT**.
# 

#    
# 

# <a id='item12'></a>
# 

#   
# 

# <a id='item13'></a>
# 

# ## Load Images
# 

# Next, we will use the standard approach of loading all images into memory and demonstrate how this approach is not efficient at all when it comes to building deep learning models for classifying images.
# 

# Let's start by reading in the negative images. First, we will use **os.scandir** to build an iterator to iterate through *./Negative* directory that contains all the images with no cracks.
# 

# In[36]:


#negative_files = os.scandir('./Negative')
negative_directory = directory +'/Negative'
negative_files = os.scandir(negative_directory)
print(negative_files)


# Then, we will grab the first file in the directory.
# 

# In[37]:


file_name = next(negative_files)
print(file_name)


# Since the directory can contain elements that are not files, we will only read the element if it is a file.
# 

# In[38]:


os.path.isfile(file_name)


# Get the image name.
# 

# In[39]:


image_name = str(file_name).split("'")[1]
print(image_name)


# Read in the image data.
# 

# In[40]:


#image_data = plt.imread('./Negative/{}'.format(image_name))
image_data = plt.imread(negative_directory+'/{}'.format(image_name))
print(image_data)


# ### **Question**: What is the dimension of a single image according to **image_data**? 
# 

# In[41]:


## You can use this cell to type your code to answer the above question
print(image_data.ndim)


# Let's view the image.
# 

# In[42]:


plt.imshow(image_data)


# Now that we are familiar with the process of reading in an image data, let's loop through all the image in the *./Negative* directory and read them all in and save them in the list **negative_images**. We will also time it to see how long it takes to read in all the images.
# 

# In[43]:


#get_ipython().run_cell_magic('time', '', '\nnegative_images = []\nfor file_name in negative_files:\n    if os.path.isfile(file_name):\n        image_name = str(file_name).split("\'")[1]\n        image_data = plt.imread(\'./Negative/{}\'.format(image_name))\n        negative_images.append(image_data)\n    \nnegative_images = np.array(negative_images)\n')
import os
import numpy as np
import matplotlib.pyplot as plt
import time

# Démarrer le chronomètre
start_time = time.time()

# Charger les images négatives
negative_images = []
for file_name in negative_files:
    if os.path.isfile(file_name):
        image_name = str(file_name).split("'")[1]
        #image_data = plt.imread('./Negative/{}'.format(image_name))
        image_data = plt.imread(negative_directory+'/{}'.format(image_name))
        negative_images.append(image_data)

negative_images = np.array(negative_images)

# Arrêter le chronomètre
end_time = time.time()
print(f"Temps d'exécution : {end_time - start_time:.2f} secondes")


# Oops! The **KERNEL DIED** due to an out-of-memory error. Since the kernel died, you may have to run the above cell to load the libraries and packages again.
# 
# Loading images into memory is definitely not the right approach when working with images as you can hit your limit on memory and other resources fairly quickly. Therefore, let's repeat the previous process but let's save the paths to the images in a variable instead of loading and saving the images themselves.
# 

# So instead of using **os.scandir**, we will use **os.listdir**.
# 

# In[44]:


#negative_images = os.listdir('./Negative')
negative_images = os.listdir(negative_directory)
print(negative_images)


# Notice how the images are not sorted, so let's call the <code>sort</code> method to sort the images.
# 

# In[45]:


negative_images.sort()
print(negative_images)


# Before we can show an image, we need to open it, which we can do using the **Image** module in the **PIL** library. So to open the first image, we run the following:
# 

# In[46]:


#image_data = Image.open('./Negative/{}'.format(negative_images[0]))
image_data = Image.open(negative_directory+'/{}'.format(negative_images[0]))


# Then to view the image, you can simply run:
# 

# In[47]:


print(image_data)


# or use the <code>imshow</code> method as follows:
# 

# In[48]:


plt.imshow(image_data)


# Let's loop through all the images in the <code>./Negative</code> directory and add save their paths.
# 

# In[49]:


#negative_images_dir = ['./Negative/{}'.format(image) for image in negative_images]
negative_images_dir = [negative_directory+'/{}'.format(image) for image in negative_images]
print(negative_images_dir)


# Let's check how many images with no cracks exist in the dataset.
# 

# In[50]:


len(negative_images_dir)


# ### Question: Show the next four images.
# 

# In[66]:


## You can use this cell to type your code to answer the above question

image_data = []

for i in range(4):
    #image_data.append(Image.open('./Negative/{}'.format(negative_images[i+1])))
    image_data.append(Image.open(negative_directory+'/{}'.format(negative_images[i+1])))

# Create a grid to display all images
fig, axes = plt.subplots(1, len(image_data), figsize=(12, 4))  # 1 row, len(image_data) columns
for ax, img in zip(axes, image_data):
    ax.imshow(img)
    ax.axis('off')  # Hide axes

plt.show()    


# **Your turn**: Save the paths to all the images in the *./Positive* directory in a list called **positive_images_dir**. Make sure to sort the paths.
# 

# In[56]:


## Type your answer here
#positive_images = os.listdir('./Positive')
positive_directory = directory + '/Positive' #AV
positive_images = os.listdir(positive_directory)
positive_images.sort()
#positive_images_dir = ['./Positive/{}'.format(image) for image in positive_images]
positive_images_dir = [positive_directory+'/{}'.format(image) for image in positive_images]
print(positive_images_dir)


# ### Question: How many images of cracked concrete exist in the *./Positive* directory?
# 

# In[63]:


## You can use this cell to type your code to answer the above question
len(positive_images_dir)
#img = Image.open('./Positive/{}'.format(positive_images[0]))  # Open the first image
img = Image.open(positive_directory+'/{}'.format(positive_images[0]))  # Open the first image
width, height = img.size
print("Width:", width)
print("Height:", height)


# ### Question: Show the first four images with cracked concrete.
# 

# In[58]:


## You can use this cell to type your code to answer the above question

image_data = []

for i in range(4):
    #image_data.append(Image.open('./Positive/{}'.format(positive_images[i])))
    image_data.append(Image.open(positive_directory+'/{}'.format(positive_images[i])))

# Create a grid to display all images
fig, axes = plt.subplots(1, len(image_data), figsize=(12, 4))  # 1 row, len(image_data) columns
for ax, img in zip(axes, image_data):
    ax.imshow(img)
    ax.axis('off')  # Hide axes

plt.show()    






#  
# 

# ### Thank you for completing this lab!
# 
# This notebook was created by Alex Aklson. I hope you found this lab interesting and educational.
# 

# This notebook is part of a course on **Coursera** called *AI Capstone Project with Deep Learning*. If you accessed this notebook outside the course, you can take this course online by clicking [here](https://cocl.us/DL0321EN_Coursera_Week1_LAB1).
# 

# <h2>About the Authors:</h2> 
# 
# <a href="https://www.linkedin.com/in/joseph-s-50398b136/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkDL0321ENSkillsNetwork951-2022-01-01">Joseph Santarcangelo</a> has a PhD in Electrical Engineering, his research focused on using machine learning, signal processing, and computer vision to determine how videos impact human cognition. Joseph has been working for IBM since he completed his PhD.
# 

#  [Alex Aklson](https://www.linkedin.com/in/aklson/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkDL0321ENSkillsNetwork951-2022-01-01). Ph.D., is a data scientist in the Digital Business Group at IBM Canada. Alex has been intensively involved in many exciting data science projects such as designing a smart system that could detect the onset of dementia in older adults using longitudinal trajectories of walking speed and home activity. Before joining IBM, Alex worked as a data scientist at Datascope Analytics, a data science consulting firm in Chicago, IL, where he designed solutions and products using a human-centred, data-driven approach. Alex received his Ph.D. in Biomedical Engineering from the University of Toronto.
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
