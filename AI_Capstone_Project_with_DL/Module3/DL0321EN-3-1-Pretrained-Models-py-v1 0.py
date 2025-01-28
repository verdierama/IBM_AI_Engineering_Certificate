#!/usr/bin/env python
# coding: utf-8

# <a href="https://cognitiveclass.ai"><img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DL0321EN-SkillsNetwork/image/IDSN-logo.png" width="400"> </a>
# 
# <h1 align=center><font size = 5>Pre-Trained Models</font></h1>
# 

# ## Objective
# 

# In this lab, you will learn how to leverage pre-trained models to build image classifiers instead of building a model from scratch.
# 

# ## Table of Contents
# 
# <div class="alert alert-block alert-info" style="margin-top: 20px">
# 
# <font size = 3> 
#     
# 1. <a href="#item31">Import Libraries and Packages</a>
# 2. <a href="#item32">Download Data</a>  
# 3. <a href="#item33">Define Global Constants</a>  
# 4. <a href="#item34">Construct ImageDataGenerator Instances</a>  
# 5. <a href="#item35">Compile and Fit Model</a>
# 
# </font>
#     
# </div>
# 

#    
# 

# <a id='item31'></a>
# 

# ## Import Libraries and Packages
# 

# Let's start the lab by importing the libraries that we will be using in this lab. First we will need the library that helps us to import the data.
# 

# In[1]:


import skillsnetwork 


# First, we will import the ImageDataGenerator module since we will be leveraging it to train our model in batches.
# 

# In[2]:
#pip install tensorflow==2.16.1 with keras embedded
import sys
print(f"Version de Python: {sys.version}")
import tensorflow as tf
print(f"TensorFlow is imported: {tf.__name__}")
print(f"TensorFlow version: {tf.__version__}")
#from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In this lab, we will be using the Keras library to build an image classifier, so let's download the Keras library.
# 

# In[3]:


import keras
from keras.models import Sequential
from keras.layers import Dense


# Finally, we will be leveraging the ResNet50 model to build our classifier, so let's download it as well.
# 

# In[4]:


from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input


# <a id='item32'></a>
# 

# ## Download Data
# 

# In this section, you are going to download the data from IBM object storage using **skillsnetwork.prepare** command. skillsnetwork.prepare is a command that's used to download a zip file, unzip it and store it in a specified directory.
# 

# In[5]:


## get the data
#await skillsnetwork.prepare("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DL0321EN-SkillsNetwork/concrete_data_week3.zip", overwrite=True)
import os
import asyncio
from skillsnetwork import prepare

directory="./concrete_data_week3"
os.makedirs(directory, exist_ok=True)
if not os.listdir(directory):
    async def download_data():
        await prepare(
            "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DL0321EN-SkillsNetwork/concrete_data_week3.zip",
            overwrite=True
        )
    # Appel direct avec asyncio.run()
    asyncio.run(download_data())


# Now, you should see the folder *concrete_data_week3* appear in the left pane. If you open this folder by double-clicking on it, you will find that it contains two folders: *train* and *valid*. And if you explore these folders, you will find that each contains two subfolders: *positive* and *negative*. These are the same folders that we saw in the labs in the previous modules of this course, where *negative* is the negative class and it represents the concrete images with no cracks and *positive* is the positive class and it represents the concrete images with cracks.
# 

# **Important Note**: There are thousands and thousands of images in each folder, so please don't attempt to double click on the *negative* and *positive* folders. This may consume all of your memory and you may end up with a **50** error. So please **DO NOT DO IT**.
# 

# <a id='item33'></a>
# 

# ## Define Global Constants
# 

# Here, we will define constants that we will be using throughout the rest of the lab. 
# 
# 1. We are obviously dealing with two classes, so *num_classes* is 2. 
# 2. The ResNet50 model was built and trained using images of size (224 x 224). Therefore, we will have to resize our images from (227 x 227) to (224 x 224).
# 3. We will training and validating the model using batches of 100 images.
# 

# In[6]:


num_classes = 2

image_resize = 224

batch_size_training = 100
batch_size_validation = 100


# <a id='item34'></a>
# 

# ## Construct ImageDataGenerator Instances
# 

# In order to instantiate an ImageDataGenerator instance, we will set the **preprocessing_function** argument to *preprocess_input* which we imported from **keras.applications.resnet50** in order to preprocess our images the same way the images used to train ResNet50 model were processed.
# 

# In[7]:


data_generator = ImageDataGenerator(
    preprocessing_function=preprocess_input,
)


# Next, we will use the *flow_from_directory* method to get the training images as follows:
# 

# In[8]:


train_generator = data_generator.flow_from_directory(
    'concrete_data_week3/train',
    target_size=(image_resize, image_resize),
    batch_size=batch_size_training,
    class_mode='categorical')


# **Note**: in this lab, we will be using the full data-set of 30,000 images for training and validation.
# 

# **Your Turn**: Use the *flow_from_directory* method to get the validation images and assign the result to **validation_generator**.
# 

# In[9]:


## Type your answer here
validation_generator = data_generator.flow_from_directory(
    'concrete_data_week3/valid',
    target_size=(image_resize, image_resize),
    batch_size=batch_size_validation,
    class_mode='categorical')



# Double-click __here__ for the solution.
# <!-- The correct answer is:
# validation_generator = data_generator.flow_from_directory(
#     'concrete_data_week3/valid',
#     target_size=(image_resize, image_resize),
#     batch_size=batch_size_validation,
#     class_mode='categorical')
# -->
# 
# 

# <a id='item35'></a>
# 

# ## Build, Compile and Fit Model
# 

# In this section, we will start building our model. We will use the Sequential model class from Keras.
# 

# In[10]:


model = Sequential()


# Next, we will add the ResNet50 pre-trained model to out model. However, note that we don't want to include the top layer or the output layer of the pre-trained model. We actually want to define our own output layer and train it so that it is optimized for our image dataset. In order to leave out the output layer of the pre-trained model, we will use the argument *include_top* and set it to **False**.
# 

# In[11]:


model.add(ResNet50(
    include_top=False,
    pooling='avg',
    weights='imagenet',
    ))


# Then, we will define our output layer as a **Dense** layer, that consists of two nodes and uses the **Softmax** function as the activation function.
# 

# In[12]:


model.add(Dense(num_classes, activation='softmax'))


# You can access the model's layers using the *layers* attribute of our model object. 
# 

# In[13]:


print(model.layers)


# You can see that our model is composed of two sets of layers. The first set is the layers pertaining to ResNet50 and the second set is a single layer, which is our Dense layer that we defined above.
# 

# You can access the ResNet50 layers by running the following:
# 

# In[14]:


print(model.layers[0].layers)


# Since the ResNet50 model has already been trained, then we want to tell our model not to bother with training the ResNet part, but to train only our dense output layer. To do that, we run the following.
# 

# In[15]:


model.layers[0].trainable = False


# And now using the *summary* attribute of the model, we can see how many parameters we will need to optimize in order to train the output layer.
# 

# In[16]:


print(model.summary())


# Next we compile our model using the **adam** optimizer.
# 

# In[17]:


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Before we are able to start the training process, with an ImageDataGenerator, we will need to define how many steps compose an epoch. Typically, that is the number of images divided by the batch size. Therefore, we define our steps per epoch as follows:
# 

# In[18]:


steps_per_epoch_training = len(train_generator)
steps_per_epoch_validation = len(validation_generator)
num_epochs = 2


# Finally, we are ready to start training our model. Unlike a conventional deep learning training were data is not streamed from a directory, with an ImageDataGenerator where data is augmented in batches, we use the **fit_generator** method.
# 

# In[ ]:


#fit_history = model.fit_generator(
fit_history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch_training,
    epochs=num_epochs,
    validation_data=validation_generator,
    validation_steps=steps_per_epoch_validation,
    verbose=1,
)


# Now that the model is trained, you are ready to start using it to classify images.
# 

# Since training can take a long time when building deep learning models, it is always a good idea to save your model once the training is complete if you believe you will be using the model again later. You will be using this model in the next module, so go ahead and save your model.
# 

# In[ ]:


model.save('classifier_resnet_model.h5')


# Now, you should see the model file *classifier_resnet_model.h5* apprear in the left directory pane.
# 

# ### Thank you for completing this lab!
# 
# This notebook was created by Alex Aklson. I hope you found this lab interesting and educational.
# 

# This notebook is part of a course on **Coursera** called *AI Capstone Project with Deep Learning*. If you accessed this notebook outside the course, you can take this course online by clicking [here](https://cocl.us/DL0321EN_Coursera_Week3_LAB1).
# 

# 
# ## Change Log
# 
# |  Date (YYYY-MM-DD) |  Version | Changed By  |  Change Description |
# |---|---|---|---|
# | 2020-09-18  | 2.0  | Shubham  |  Migrated Lab to Markdown and added to course repo in GitLab |
# | 2023-01-03  | 3.0  | Artem |  Updated the file import section|
# 
# 

# <hr>
# 
# Copyright &copy; 2020 [IBM Developer Skills Network](https://cognitiveclass.ai/?utm_source=bducopyrightlink&utm_medium=dswb&utm_campaign=bdu). This notebook and its source code are released under the terms of the [MIT License](https://bigdatauniversity.com/mit-license/).
# 
