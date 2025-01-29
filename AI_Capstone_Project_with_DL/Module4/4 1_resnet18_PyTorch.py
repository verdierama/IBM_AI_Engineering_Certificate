#!/usr/bin/env python
# coding: utf-8

# <a href="http://cocl.us/pytorch_link_top">
#     <img src="https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0110EN/notebook_images%20/Pytochtop.png" width="750" alt="IBM Product ">
# </a> 
# 

# <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DL0321EN-SkillsNetwork/image/IDSN-logo.png" width="200" alt="cognitiveclass.ai logo">
# 

# <h1><h1>Pre-trained-Models with PyTorch </h1>
# 

# In this lab, you will use pre-trained models to classify between the negative and positive samples; you will be provided with the dataset object. The particular pre-trained model will be resnet18; you will have three questions: 
# <ul>
# <li>change the output layer</li>
# <li> train the model</li> 
# <li>  identify  several  misclassified samples</li> 
#  </ul>
# You will take several screenshots of your work and share your notebook. 
# 

# <h2>Table of Contents</h2>
# 

# <div class="alert alert-block alert-info" style="margin-top: 20px">
# 
# 
# <ul>
#     <li><a href="#download_data"> Download Data</a></li>
#     <li><a href="#auxiliary"> Imports and Auxiliary Functions </a></li>
#     <li><a href="#data_class"> Dataset Class</a></li>
#     <li><a href="#Question_1">Question 1</a></li>
#     <li><a href="#Question_2">Question 2</a></li>
#     <li><a href="#Question_3">Question 3</a></li>
# </ul>
# <p>Estimated Time Needed: <strong>120 min</strong></p>
#  </div>
# <hr>
# 

# <h2 id="download_data">Download Data</h2>
# 

# Download the dataset and unzip the files in your data directory, unlike the other labs, all the data will be deleted after you close  the lab, this may take some time:
# 

# In[1]:


#get_ipython().system('wget https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0321EN/data/images/Positive_tensors.zip')
import os
import asyncio
from skillsnetwork import prepare

zip_file_path = "Positive_tensors.zip"
if not os.path.exists(zip_file_path):
    async def download_data():
        await prepare(
            "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0321EN/data/images/Positive_tensors.zip",
            overwrite=True
        )
    # Appel direct avec asyncio.run()
    asyncio.run(download_data())

# In[15]:


#get_ipython().system('unzip -n -q Positive_tensors.zip')
import zipfile


output_dir = "./"

# Créer le répertoire de destination s'il n'existe pas
#os.makedirs(output_dir, exist_ok=True)

# Ouvrir le fichier ZIP
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    for file_name in zip_ref.namelist():
        # Construire le chemin complet du fichier de destination
        extracted_path = os.path.join(output_dir, file_name)

        # Vérifier si le fichier existe déjà
        if not os.path.exists(extracted_path):
            zip_ref.extract(file_name, output_dir)
        else:
            print(f"Le fichier existe déjà, ignoré : {extracted_path}")

print(f"Les fichiers ont été extraits dans : {output_dir}")


# In[ ]:


#get_ipython().system("find 'Positive_tensors' -maxdepth 1 -type f | wc -l")
#get_ipython().system("find 'Negative_tensors' -maxdepth 1 -type f | wc -l")
#! wget https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0321EN/data/images/Negative_tensors.zip

zip_file_path = "Negative_tensors.zip"
if not os.path.exists(zip_file_path):
    async def download_data():
        await prepare(
            "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0321EN/data/images/Negative_tensors.zip",
            overwrite=True
        )
    # Appel direct avec asyncio.run()
    asyncio.run(download_data())

#get_ipython().system('unzip -n -q Negative_tensors.zip')
#output_dir = "Negative_tensors"

# Créer le répertoire de destination s'il n'existe pas
os.makedirs(output_dir, exist_ok=True)

# Ouvrir le fichier ZIP
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    for file_name in zip_ref.namelist():
        # Construire le chemin complet du fichier de destination
        extracted_path = os.path.join(output_dir, file_name)

        # Vérifier si le fichier existe déjà
        if not os.path.exists(extracted_path):
            zip_ref.extract(file_name, output_dir)
        else:
            print(f"Le fichier existe déjà, ignoré : {extracted_path}")

print(f"Les fichiers ont été extraits dans : {output_dir}")



# Compter les fichiers dans 'Positive_tensors'
positive_dir = 'Positive_tensors'
positive_file_count = len([f for f in os.listdir(positive_dir) if os.path.isfile(os.path.join(positive_dir, f))])
print(f"Nombre de fichiers dans '{positive_dir}': {positive_file_count}")

# Compter les fichiers dans 'Negative_tensors'
negative_dir = 'Negative_tensors'
negative_file_count = len([f for f in os.listdir(negative_dir) if os.path.isfile(os.path.join(negative_dir, f))])
print(f"Nombre de fichiers dans '{negative_dir}': {negative_file_count}")


# We will install torchvision:
# 

# In[4]:


#get_ipython().system('pip install torchvision')


# <h2 id="auxiliary">Imports and Auxiliary Functions</h2>
# 

# The following are the libraries we are going to use for this lab. The <code>torch.manual_seed()</code> is for forcing the random function to give the same number every time we try to recompile it.
# 

# In[5]:


# These are the libraries will be used for this lab.
import torchvision.models as models
from PIL import Image
import pandas
from torchvision import transforms
import torch.nn as nn
import time
import torch 
import matplotlib.pylab as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
import h5py
import os
import glob
torch.manual_seed(0)


# In[6]:


from matplotlib.pyplot import imshow
import matplotlib.pylab as plt
from PIL import Image
import pandas as pd
import os


# <!--Empty Space for separating topics-->
# 

# <h2 id="data_class">Dataset Class</h2>
# 

#  This dataset class is essentially the same dataset you build in the previous section, but to speed things up, we are going to use tensors instead of jpeg images. Therefor for each iteration, you will skip the reshape step, conversion step to tensors and normalization step.
# 

# In[7]:


# Create your own dataset object

class Dataset(Dataset):

    # Constructor
    def __init__(self,transform=None,train=True):
        directory="./"
        positive="Positive_tensors"
        negative='Negative_tensors'

        positive_file_path=os.path.join(directory,positive)
        negative_file_path=os.path.join(directory,negative)
        positive_files=[os.path.join(positive_file_path,file) for file in os.listdir(positive_file_path) if file.endswith(".pt")]
        negative_files=[os.path.join(negative_file_path,file) for file in os.listdir(negative_file_path) if file.endswith(".pt")]
        number_of_samples=len(positive_files)+len(negative_files)
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
               
        image=torch.load(self.all_files[idx])
        y=self.Y[idx]
                  
        # If there is any transform method, apply it onto the image
        if self.transform:
            image = self.transform(image)

        return image, y
    
print("done")


# We create two dataset objects, one for the training data and one for the validation data.
# 

# In[8]:


train_dataset = Dataset(train=True)
validation_dataset = Dataset(train=False)
print("done")


# <h2 id="Question_1">Question 1</h2>
# 

# <b>Prepare a pre-trained resnet18 model :</b>
# 

# <b>Step 1</b>: Load the pre-trained model <code>resnet18</code> Set the parameter <code>pretrained</code> to true:
# 

# In[ ]:


# Step 1: Load the pre-trained model resnet18

# Type your code here
import torchvision.models as models

# Load the pre-trained ResNet-18 model
model = models.resnet18(pretrained=True)

# Print the model architecture to confirm it is loaded
print(model)

# <b>Step 2</b>: Set the attribute <code>requires_grad</code> to <code>False</code>. As a result, the parameters will not be affected by training.
# 

# In[ ]:


# Step 2: Set the parameter cannot be trained for the pre-trained model


# Type your code here
# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Print to verify
for name, param in model.named_parameters():
    print(name, param.requires_grad)  # Should print "False" for all


# <code>resnet18</code> is used to classify 1000 different objects; as a result, the last layer has 1000 outputs.  The 512 inputs come from the fact that the previously hidden layer has 512 outputs. 
# 

# <b>Step 3</b>: Replace the output layer <code>model.fc</code> of the neural network with a <code>nn.Linear</code> object, to classify 2 different classes. For the parameters <code>in_features </code> remember the last hidden layer has 512 neurons.
# 
num_classes = 2  # Binary classification (e.g., class 0 and class 1)
model.fc = nn.Linear(in_features=512, out_features=num_classes)
# In[ ]:





# Print out the model in order to show whether you get the correct answer.<br> <b>(Your peer reviewer is going to mark based on what you print here.)</b>
# 

# In[ ]:


print(model)


# <h2 id="Question_2">Question 2: Train the Model</h2>
# 

# In this question you will train your, model:
# 

# <b>Step 1</b>: Create a cross entropy criterion function 
# 

# In[ ]:


# Step 1: Create the loss function

# Type your code here
# Define CrossEntropyLoss (already implemented in PyTorch)
criterion = nn.CrossEntropyLoss()

# <b>Step 2</b>: Create a training loader and validation loader object, the batch size should have 100 samples each.
# 

# In[ ]:
from torchvision import transforms

transform = transforms.Compose([
#    transforms.ToTensor(),
    transforms.Resize((224, 224)),  # Resize to match the model's expected input size
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])
dataset_train=Dataset(transform=transform, train=True)
dataset_val=Dataset(transform=transform, train=False)
batch_size = 100
train_loader = torch.utils.data.DataLoader(dataset=dataset_train, batch_size=batch_size)
validation_loader = torch.utils.data.DataLoader(dataset=dataset_val, batch_size=batch_size)




# <b>Step 3</b>: Use the following optimizer to minimize the loss 
# 

# In[ ]:


optimizer = torch.optim.Adam([parameters  for parameters in model.parameters() if parameters.requires_grad],lr=0.001)


# <!--Empty Space for separating topics-->
# 

# **Complete the following code to calculate  the accuracy on the validation data for one epoch; this should take about 45 minutes. Make sure you calculate the accuracy on the validation data.**
# 

# In[ ]:


n_epochs = 1
loss_list = []
accuracy_list = []
correct = 0
N_test = len(validation_dataset)
N_train = len(train_dataset)
start_time = time.time()
# n_epochs
cnt = 0
Loss = 0
start_time = time.time()
for epoch in range(n_epochs):
    for x, y in train_loader:
        print(cnt)
        cnt = cnt + 1
        model.train()
        # clear gradient
        optimizer.zero_grad()
        # make a prediction
        y_pred = model(x)
        # calculate loss
        loss = criterion(y_pred, y)
        # calculate gradients of parameters
        loss.backward()
        # update parameters
        optimizer.step()

        loss_list.append(loss.data)

    # Validation loop
    correct = 0
    misclassified_samples = []  # To store the first four misclassified samples
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation
        for idx, (x_test, y_test) in enumerate(validation_loader):
            print(cnt)
            cnt = cnt + 1
            # Make a prediction
            y_pred = model(x_test)

            # Find max (for classification, we use the index of the max logit as the predicted class)
            _, predicted = torch.max(y_pred, 1)

            # Calculate misclassified samples in the mini-batch
            correct += (predicted == y_test).sum().item()

            # Compare predictions with ground truth
            for i in range(len(y_test)):
                if predicted[i].item() != y_test[i].item():  # Misclassified sample
                    misclassified_samples.append((idx * validation_loader.batch_size + i, y_test[i].item(), predicted[i].item()))

    accuracy = correct / N_test

# <b>Print out the Accuracy and plot the loss stored in the list <code>loss_list</code> for every iteration and take a screen shot.</b>
# 

# In[ ]:


print("accuracy = ",accuracy)

for sample in misclassified_samples[:4]:
    index, true_label, predicted_label = sample
    print(f"Sample index: {index}, True label: {true_label}, Predicted label: {predicted_label}")


# In[ ]:


plt.plot(loss_list)
plt.xlabel("iteration")
plt.ylabel("loss")
plt.show()


# <h2 id="Question_3">Question 3:Find the misclassified samples</h2> 
# 

# <b>Identify the first four misclassified samples using the validation data:</b>
# 

# In[ ]:





# <a href="https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/share-notebooks.html?utm_source=Exinfluencer&utm_content=000026UJ&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkDL0321ENSkillsNetwork951-2022-01-01&utm_medium=Exinfluencer&utm_term=10006555"> CLICK HERE </a> Click here to see how to share your notebook.
# 

# <h2>About the Authors:</h2> 
# 
# <a href="https://www.linkedin.com/in/joseph-s-50398b136/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkDL0321ENSkillsNetwork951-2022-01-01">Joseph Santarcangelo</a> has a PhD in Electrical Engineering, his research focused on using machine learning, signal processing, and computer vision to determine how videos impact human cognition. Joseph has been working for IBM since he completed his PhD.
# 

# 
# ## Change Log
# 
# |  Date (YYYY-MM-DD) |  Version | Changed By  |  Change Description |
# |---|---|---|---|
# | 2020-09-21  | 2.0  | Shubham  |  Migrated Lab to Markdown and added to course repo in GitLab |
# 
# 
# 
# <hr>
# 
# ## <h3 align="center"> © IBM Corporation 2020. All rights reserved. <h3/>
# 

# Copyright &copy; 2018 <a href="cognitiveclass.ai?utm_source=bducopyrightlink&utm_medium=dswb&utm_campaign=bdu">cognitiveclass.ai</a>. This notebook and its source code are released under the terms of the <a href="https://bigdatauniversity.com/mit-license/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDeveloperSkillsNetworkDL0321ENSkillsNetwork951-2022-01-01">MIT License</a>.
# 
