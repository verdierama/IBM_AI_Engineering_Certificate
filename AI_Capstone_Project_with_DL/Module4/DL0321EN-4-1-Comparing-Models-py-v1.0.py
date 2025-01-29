# CAUTION, file is the original template
#!/usr/bin/env python
# coding: utf-8

# <a href="https://cognitiveclass.ai"><img src = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/Logos/organization_logo/organization_logo.png" width = 400> </a>
# 
# <h1 align=center><font size = 5>Peer Review Final Assignment</font></h1>

# ## Introduction
# 

# In this lab, you will build an image classifier using the VGG16 pre-trained model, and you will evaluate it and compare its performance to the model we built in the last module using the ResNet50 pre-trained model. Good luck!

# ## Table of Contents
# 
# <div class="alert alert-block alert-info" style="margin-top: 20px">
# 
# <font size = 3>    
# 
# 1. <a href="#item41">Download Data 
# 2. <a href="#item42">Part 1</a>
# 3. <a href="#item43">Part 2</a>  
# 4. <a href="#item44">Part 3</a>  
# 
# </font>
#     
# </div>

#    

# <a id="item41"></a>

# ## Download Data

# Use the <code>wget</code> command to download the data for this assignment from here: https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0321EN/data/concrete_data_week4.zip

# Use the following cells to download the data.

# In[ ]:





# In[ ]:





# After you unzip the data, you fill find the data has already been divided into a train, validation, and test sets.

#   

# <a id="item42"></a>

# ## Part 1

# In this part, you will design a classifier using the VGG16 pre-trained model. Just like the ResNet50 model, you can import the model <code>VGG16</code> from <code>keras.applications</code>.

# You will essentially build your classifier as follows:
# 1. Import libraries, modules, and packages you will need. Make sure to import the *preprocess_input* function from <code>keras.applications.vgg16</code>.
# 2. Use a batch size of 100 images for both training and validation.
# 3. Construct an ImageDataGenerator for the training set and another one for the validation set. VGG16 was originally trained on 224 × 224 images, so make sure to address that when defining the ImageDataGenerator instances.
# 4. Create a sequential model using Keras. Add VGG16 model to it and dense layer.
# 5. Compile the mode using the adam optimizer and the categorical_crossentropy loss function.
# 6. Fit the model on the augmented data using the ImageDataGenerators.

# Use the following cells to create your classifier.

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





#    

# <a id="item43"></a>

# ## Part 2

# In this part, you will evaluate your deep learning models on a test data. For this part, you will need to do the following:
# 
# 1. Load your saved model that was built using the ResNet50 model. 
# 2. Construct an ImageDataGenerator for the test set. For this ImageDataGenerator instance, you only need to pass the directory of the test images, target size, and the **shuffle** parameter and set it to False.
# 3. Use the **evaluate_generator** method to evaluate your models on the test data, by passing the above ImageDataGenerator as an argument. You can learn more about **evaluate_generator** [here](https://keras.io/models/sequential/).
# 4. Print the performance of the classifier using the VGG16 pre-trained model.
# 5. Print the performance of the classifier using the ResNet pre-trained model.
# 

# Use the following cells to evaluate your models.

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





#    

# <a id="item44"></a>

# ## Part 3

# In this model, you will predict whether the images in the test data are images of cracked concrete or not. You will do the following:
# 
# 1. Use the **predict_generator** method to predict the class of the images in the test data, by passing the test data ImageDataGenerator instance defined in the previous part as an argument. You can learn more about the **predict_generator** method [here](https://keras.io/models/sequential/).
# 2. Report the class predictions of the first five images in the test set. You should print something list this:
# 
# <center>
#     <ul style="list-style-type:none">
#         <li>Positive</li>  
#         <li>Negative</li> 
#         <li>Positive</li>
#         <li>Positive</li>
#         <li>Negative</li>
#     </ul>
# </center>

# Use the following cells to make your predictions.

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





#   

# ### Thank you for completing this lab!
# 
# This notebook was created by Alex Aklson.

# This notebook is part of a course on **Coursera** called *AI Capstone Project with Deep Learning*. If you accessed this notebook outside the course, you can take this course online by clicking [here](https://cocl.us/DL0321EN_Coursera_Week4_LAB1).

# <hr>
# 
# Copyright &copy; 2020 [IBM Developer Skills Network](https://cognitiveclass.ai/?utm_source=bducopyrightlink&utm_medium=dswb&utm_campaign=bdu). This notebook and its source code are released under the terms of the [MIT License](https://bigdatauniversity.com/mit-license/).
