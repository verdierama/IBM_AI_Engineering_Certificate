#!/usr/bin/env python
# coding: utf-8

# <a href="http://cocl.us/pytorch_link_top">
#     <img src="https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0110EN/notebook_images%20/Pytochtop.png" width="750" alt="IBM Product ">
# </a> 
# 

# <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DL0321EN-SkillsNetwork/image/IDSN-logo.png" width="200" alt="cognitiveclass.ai logo">
# 

# <h1>Objective</h1><ul><li> How to create a dataset object.</li></ul> 
# 

# <h1>Data Preparation with PyTorch</h1>
# 

# <p>Crack detection has vital importance for structural health monitoring and inspection. We would like to train a network to detect Cracks, we will denote the images that contain cracks as positive and images with no cracks as negative.  In this lab you are going to have to build a dataset object. There are five questions in this lab, Including some questions that are intermediate steps to help you build the dataset object. You are going to have to remember the output for some  of the questions. </p>
# 

# <h2>Table of Contents</h2>
# 

# <div class="alert alert-block alert-info" style="margin-top: 20px">
# 
# 
# <ul>
#     <li><a href="#auxiliary"> Imports and Auxiliary Functions </a></li>
#     <li><a href="#download_data"> Download data</a></li>
#     <li><a href="#examine_files">Examine Files</a></li> 
#     <li><a href="#Question_1"><b>Question 1:find number of files</b> </a></li>
#     <li><a href="#assign_labels">Assign Labels to Images  </a></li>
#     <li><a href="#Question_2"><b>Question 2 : Assign labels to image </b> </a></li>
#     <li><a href="#split">Training  and Validation  Split </a></li>
#     <li><a href="#Question_3"><b>Question 3: Training  and Validation  Split</b> </a></li>
# <li><a href="#data_class">Create a Dataset Class </a></li>
#     <li><a href="#Question_4"><b>Question 4:Display  training dataset object</b> </a></li>
#     <li><a href="#Question_5"><b>Question 5:Display  validation dataset  object</b> </a></li>
# 
# </ul>
# <p>Estimated Time Needed: <strong>25 min</strong></p>
#  </div>
# <hr>
# 

# <h2 id="auxiliary">Imports and Auxiliary Functions</h2>
# 

# The following are the libraries we are going to use for this lab:
# 

# In[77]:


from PIL import Image
import matplotlib.pyplot as plt
import os
import glob
import torch
from torch.utils.data import Dataset
import skillsnetwork 


# We will use this function in the lab to plot:
# 

# In[78]:


def show_data(data_sample, shape = (28, 28)):
    plt.imshow(data_sample[0].numpy().reshape(shape), cmap='gray')
    plt.title('y = ' + data_sample[1])


# <h2 id="download_data">Download Data</h2>
# 

# In this section, you are going to download the data from IBM object storage using **skillsnetwork.prepare** command. <b>skillsnetwork.prepare</b> is a command that's used to download a zip file, unzip it and store it in a specified directory. Locally we store the data in the directory  **/resources/data**. 
# 

# In[79]:


#await skillsnetwork.prepare("https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0321EN/data/images/concrete_crack_images_for_classification.zip", path = "/resources/data", overwrite=True)
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




# We then download the files that contain the negative images:
# 

# <h2 id="examine_files">Examine Files </h2>
# 

# In the previous lab, we create two lists; one to hold the path to the Negative files and one to hold the path to the Positive files. This process is shown in the following few lines of code.
# 

# We can obtain the list that contains the path to the <b>negative files</b> as follows:
# 

# In[80]:


#directory="/resources/data"
negative='Negative'
negative_file_path=os.path.join(directory,negative)
negative_files=[os.path.join(negative_file_path,file) for file in  os.listdir(negative_file_path) if file.endswith(".jpg")]
negative_files.sort()
print(negative_files[0:3])


# We can obtain the list that contains the path to the <b>positive files</b> files as follows:
# 

# In[81]:


positive="Positive"
positive_file_path=os.path.join(directory,positive)
positive_files=[os.path.join(positive_file_path,file) for file in  os.listdir(positive_file_path) if file.endswith(".jpg")]
positive_files.sort()
print(positive_files[0:3])


# <h2 id="Question_1">Question 1</h2>
# <b>Find the <b>combined</b> length of the list <code>positive_files</code> and <code>negative_files</code> using the function <code>len</code> . Then assign  it to the variable <code>number_of_samples</code></b>
# 

# In[82]:


number_of_samples = len(positive_files) + len(negative_files)
print(number_of_samples)


# <h2 id="assign_labels">Assign Labels to Images </h2>
# 

# In this section we will assign a label to each image in this case we  can assign the positive images, i.e images with a crack to a value one  and the negative images i.e images with out a crack to a value of zero <b>Y</b>. First we create a tensor or vector of zeros, each element corresponds to a new sample. The length of the tensor is equal to the number of samples.
# 

# In[83]:


Y=torch.zeros([number_of_samples])


# As we are using the tensor <b>Y</b> for classification we cast it to a <code>LongTensor</code>. 
# 

# In[84]:


Y=Y.type(torch.LongTensor)
Y.type()


# With respect to each element we will set the even elements to class one and the odd elements to class zero.
# 

# In[85]:


Y[::2]=1
Y[1::2]=0


# <h2 id="Question_2">Question 2</h2>
# <b>Create a list all_files such that the even indexes contain the path to images with positive or cracked samples and the odd element contain the negative images or images with out cracks. Then use the following code to print out the first four samples.</b>
# 

# In[86]:


all_files = []
for index in range(number_of_samples//2):
    all_files.append(positive_files[index])    # Append positive file
    all_files.append(negative_files[index])    # Append negative file


# code used to print samples:
# 

# In[87]:


for y,file in zip(Y, all_files[0:4]):
    plt.imshow(Image.open(file))
    plt.title("y="+str(y.item()))
    plt.show()
    


# <h2 id="split">Training  and Validation  Split  </h2>
# When training the model we  split up our data into training and validation data. It If the variable train is set to <code>True</code>  the following lines of code will segment the  tensor <b>Y</b> such at  the first 30000 samples are used for training. If the variable train is set to <code>False</code> the remainder of the samples will be used for validation data. 
# 

# In[88]:


'''train=False

if train:
    all_files=all_files[0:30000]
    Y=Y[0:30000]

else:
    all_files=all_files[30000:]
    Y=Y[30000:]
    '''


# <h2 id="Question_3">Question 3</h2>
# Modify the above lines of code such that if the variable <code>train</code> is set to <c>True</c> the first 30000 samples of all_files are use in training. If <code>train</code> is set to <code>False</code> the remaining  samples are used for validation. In both cases reassign  the values to the variable all_files, then use the following lines of code to print out the first four validation sample images.
# 

# In[90]:


# Assuming all_files and Y are defined and contain the correct data
training_file, training_label = all_files[0:30000], Y[0:30000]
print(len(training_file))
validation_file, validation_label = all_files[30000:], Y[30000:]
print(len(validation_file))

# Display images and their corresponding labels
for y, file in zip(validation_label, validation_file[0:4]):
    print("*")
    img = Image.open(file)  # Open the image file
    plt.imshow(img)
    plt.title("y=" + str(y.item()))  # Display the label
    plt.show()


# Just a note the images printed out in question two are the first four training samples.
# 

# <h2 id="data_class">Create a Dataset Class</h2>
# 

# In this section, we will use the previous code to build a dataset class. 
# 

# 
# Complete the code to build a Dataset class <code>dataset</code>. As before, make sure the even samples are positive, and the odd samples are negative.  If the parameter <code>train</code> is set to <code>True</code>, use the first 30 000  samples as training data; otherwise, the remaining samples will be used as validation data.  
# 

# In[93]:


class Dataset(Dataset):

    # Constructor
    def __init__(self,transform=None,train=True):
#        directory="/resources/data"
        directory="resources/data"
        positive="Positive"
        negative="Negative"

        positive_file_path=os.path.join(directory,positive)
        negative_file_path=os.path.join(directory,negative)
        positive_files=[os.path.join(positive_file_path,file) for file in  os.listdir(positive_file_path) if file.endswith(".jpg")]
        positive_files.sort()
        negative_files=[os.path.join(negative_file_path,file) for file in  os.listdir(negative_file_path) if file.endswith(".jpg")]
        negative_files.sort()

        self.all_files=[None]*number_of_samples
        self.all_files[::2]=positive_files
        self.all_files[1::2]=negative_files 
        # The transform is goint to be used on image
        self.transform = transform
        #torch.LongTensor
        self.Y=torch.zeros([number_of_samples]).type(torch.LongTensor)
        self.Y[::2]=1
        self.Y[1::2]=0
        
        if train:
            self.all_files=self.all_files[0:30000]
            self.Y=self.Y[0:30000]
            self.len=len(self.all_files)
        else:
            self.all_files=self.all_files[30000:]
            self.Y=self.Y[30000:]
            self.len=len(self.all_files)
       
    # Get the length
    def __len__(self):
        return self.len
    
    # Getter
    def __getitem__(self, idx):
        image=Image.open(self.all_files[idx])
        y=self.Y[idx]
       
    # If there is any transform method, apply it onto the image
        if self.transform:
            image = self.transform(image)

        return image, y


# <h2 id="Question_4">Question 4</h2>
# <b>Create a Dataset object <code>dataset</code> for the training data, use the following lines of code to print out sample the 10th and  sample 100 (remember zero indexing)  </b>
# 

# In[103]:


dataset = Dataset(train=True)
samples = {9,100}

for sample  in samples:
    image, label = dataset[sample] 
    plt.imshow(dataset[sample][0])
    plt.xlabel("y="+str(dataset[sample][1].item()))
    plt.title("training data, sample {}".format(int(sample)+1))
    plt.show()
    


# We now have all the tools to create a list with the path to each image file.  We use a List Comprehensions  to make the code more compact. We assign it to the variable <code>negative_files<code> , sort it in and display the first three elements:
# 

# <h2 id="Question_5">Question 5</h2>
# <b>Create a Dataset object <code>dataset</code> for the validation  data, use the following lines of code to print out the 16 th and  sample 103 (remember zero indexing)   </b>
# 

# In[104]:


dataset = Dataset(train=False)

samples = {14,101}

for sample  in samples:
    plt.imshow(dataset[sample][0])
    plt.xlabel("y="+str(dataset[sample][1].item()))
    plt.title("validation data, sample {}".format(int(sample)+1))
    plt.show()


# <h2>About the Authors:</h2>
#  <a href=\"https://www.linkedin.com/in/joseph-s-50398b136/\">Joseph Santarcangelo</a> has a PhD in Electrical Engineering, his research focused on using machine learning, signal processing, and computer vision to determine how videos impact human cognition. Joseph has been working for IBM since he completed his PhD.
# 

# 
# ## Change Log
# 
# |  Date (YYYY-MM-DD) |  Version | Changed By  |  Change Description |
# |---|---|---|---|
# | 2020-09-18  | 2.0  | Shubham  |  Migrated Lab to Markdown and added to course repo in GitLab |
# 
# 

# 

# Copyright &copy; 2018 <a href="cognitiveclass.ai">cognitiveclass.ai</a>. This notebook and its source code are released under the terms of the <a href=\"https://bigdatauniversity.com/mit-license/\">MIT License</a>
# 
