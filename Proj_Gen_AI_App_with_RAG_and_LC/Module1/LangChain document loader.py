#!/usr/bin/env python
# coding: utf-8

# <p style="text-align:center">
#     <a href="https://skills.network" target="_blank">
#     <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/assets/logos/SN_web_lightmode.png" width="200" alt="Skills Network Logo"  />
#     </a>
# </p>
# 

# # Load Documents Using LangChain for Different Sources 
# 

# Estimated time needed: **20** minutes
# 

# Imagine you are working as a data scientist at a consulting firm, and you've been tasked with analyzing documents from multiple clients. Each client provides their data in different formats: some in PDFs, others in Word documents, CSV files, or even HTML webpages. Manually loading and parsing each document type is not only time-consuming but also prone to errors. Your goal is to streamline this process, making it efficient and error-free.
# 
# To achieve this, you'll use LangChain’s powerful document loaders. These loaders allow you to read and convert various file formats into a unified document structure that can be easily processed. For example, you'll load client policy documents from text files, financial reports from PDFs, marketing strategies from Word documents, and product reviews from JSON files. By the end of this lab, you will have a robust pipeline that can handle any new file formats clients might send, saving you valuable time and effort.
# 

# <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/Hvf-jk8b5Fs-E_E4AJyEow/loader.png" width="50%" alt="indexing"/>
# 

# In this lab, you will explore how to use various loaders provided by LangChain to load and process data from different file formats. These loaders simplify the task of reading and converting files into a document format that can be processed downstream. By the end of this lab, you will be able to efficiently load text, PDF, Markdown, JSON, CSV, DOCX, and other file formats into a unified format, allowing for seamless data analysis and manipulation for LLM applications.
# 
# (Note: In this lab, we just introduced several commonly used file format loaders. LangChain provides more document loaders for various document formats [here](https://python.langchain.com/v0.2/docs/integrations/document_loaders/).)
# 

# ## __Table of Contents__
# 
# <ol>
#     <li><a href="#Objectives">Objectives</a></li>
#     <li>
#         <a href="#Setup">Setup</a>
#         <ol>
#             <li><a href="#Installing-required-libraries">Installing required libraries</a></li>
#             <li><a href="#Importing-Required-Libraries">Importing required libraries</a></li>
#         </ol>
#     </li>
#     <li><a href="#Load-from-TXT-files">Load from TXT files</a></li>
#     <li><a href="#Load-from-PDF-files">Load from PDF files</a></li>
#     <li><a href="#Load-from-Markdown-files">Load from Markdown files</a></li>
#     <li><a href="#Load-from-JSON-files">Load from JSON files</a></li>
#     <li><a href="#Load-from-CSV-files">Load from CSV files</a></li>
#     <li><a href="#Load-from-URL/Website-files">Load from URL/Webpage files</a></li>
#     <li><a href="#Load-from-WORD-files">Load from WORD files</a></li>
#     <li><a href="#Load-from-Unstructured-Files">Load from Unstructured Files</a></li>
# </ol>
# 
# <a href="#Exercises">Exercises</a>
# <ol>
#     <li><a href="#Exercise-1---Try-to-use-other-PDF-loaders">Exercise 1 - Try to use other PDF loaders</a></li>
#     <li><a href="#Exercise-2---Load-from-Arxiv">Exercise 2 - Load from Arxiv</a></li>
# </ol>
# 

# ## Objectives
# 
# After completing this lab you will be able to:
# 
#  - Understand how to use `TextLoader` to load text files.
#  - Learn how to load PDFs using `PyPDFLoader` and `PyMuPDFLoader`.
#  - Use `UnstructuredMarkdownLoader` to load Markdown files.
#  - Load JSON files with `JSONLoader` using jq schemas.
#  - Process CSV files with `CSVLoader` and `UnstructuredCSVLoader`.
#  - Load Webpage content using `WebBaseLoader`.
#  - Load Word documents using `Docx2txtLoader`.
#  - Utilize `UnstructuredFileLoader` for various file types.
# 

# ----
# 

# ## Setup
# 

# ### Installing required libraries
# 
# The following required libraries are __not__ preinstalled in the Skills Network Labs environment. __You must run the following cell__ to install them:
# 
# **Note:** We are pinning the version here to specify the version. It's recommended that you do this as well. Even if the library is updated in the future, the installed library could still support this lab work.
# 
# This might take approximately 1 minute. 
# 
# As we use `%%capture` to capture the installation, you won't see the output process. But after the installation completes, you will see a number beside the cell.
# 

# In[55]:


#get_ipython().run_cell_magic('capture', '', '#After executing the cell,please RESTART the kernel and run all the cells.\n!pip install --user "langchain-community==0.2.1"\n!pip install --user "pypdf==4.2.0"\n!pip install --user "PyMuPDF==1.24.5"\n!pip install --user "unstructured==0.14.8"\n!pip install --user "markdown==3.6"\n!pip install --user  "jq==1.7.0"\n!pip install --user "pandas==2.2.2"\n!pip install --user "docx2txt==0.8"\n!pip install --user "requests==2.32.3"\n!pip install --user "beautifulsoup4==4.12.3"\n!pip install --user "nltk==3.8.0"\n')


# After you install the libraries, restart your kernel. You can do that by clicking the **Restart the kernel** icon.
# 
# <p style="text-align:left">
#     <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/1_Bd_EvpEzLH9BbxRXXUGQ/screenshot-to-replace.png" width="50%"/>
#     </a>
# </p>
# 
# 

# ### Importing Required Libraries
# 
# _We recommend you import all required libraries in one place (here):_
# 

# In[1]:


# You can also use this section to suppress warnings generated by your code:

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

from pprint import pprint
import json
from pathlib import Path
import nltk
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.document_loaders import JSONLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.csv_loader import UnstructuredCSVLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import UnstructuredFileLoader
import pypdfium2



nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')


# ### Load from TXT files
# 

# The `TextLoader` is a tool designed to load textual data from various sources.
# 
# It is the simplest loader, reading a file as text and placing all the content into a single document.
# 

# We have prepared a .txt file for you to load. First, we need to download it from a remote source.
# 

# We have prepared a .txt file for you to load. First, we need to download it from a remote source.
# 

# In[2]:


#get_ipython().system('wget "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/Ec5f3KYU1CpbKRp1whFLZw/new-Policies.txt"')
import requests
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/Ec5f3KYU1CpbKRp1whFLZw/new-Policies.txt"
output_file = "new-Policies.txt"

# Télécharger le fichier
response = requests.get(url)
if response.status_code == 200:
    with open(output_file, 'wb') as file:
        file.write(response.content)
else:
    print(f"Erreur : Impossible de télécharger le fichier (code {response.status_code})")

# Next, we will use the `TextLoader` class to load the file.
# 

# In[3]:


loader = TextLoader("new-Policies.txt")
print(loader)


# Here, we use the `load` method to load the data as documents.
# 

# In[4]:


data = loader.load()


# Let's present the entire data (document) here.
# 
# This is a `document` object that includes `page_content` and `metadata` attributes.
# 

# In[5]:


print(data)


# We can also use the `pprint` function to print the first 1000 characters of the `page_content` here.
# 

# In[6]:


pprint(data[0].page_content[:1000])


# ### Load from PDF files
# 

# Sometimes, we may have files in PDF format that we want to load for processing.
# 
# LangChain provides several classes for loading PDFs. Here, we introduce two classes: `PyPDFLoader` and `PyMuPDFLoader`.
# 

# #### PyPDFLoader
# 

# Load the PDF using `PyPDFLoader` into an array of documents, where each document contains the page content and metadata with the page number.
# 

# In[7]:


pdf_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/Q81D33CdRLK6LswuQrANQQ/instructlab.pdf"

loader = PyPDFLoader(pdf_url)

pages = loader.load_and_split()


# Display the first page of the PDF.
# 

# In[8]:


print(pages[0])


# Display the first three pages of the PDF.
# 

# In[9]:


for p,page in enumerate(pages[0:3]):
    print(f"page number {p+1}")
    print(page)


# #### PyMuPDFLoader
# 

# `PyMuPDFLoader` is the fastest of the PDF parsing options. It provides detailed metadata about the PDF and its pages, and returns one document per page.
# 

# In[10]:


loader = PyMuPDFLoader(pdf_url)
print(loader)


# In[11]:


data = loader.load()


# In[12]:


print(data[0])


# The `metadata` attribute reveals that `PyMuPDFLoader` provides more detailed metadata information than `PyPDFLoader`.
# 

# ### Load from Markdown files
# 

# Sometimes, our file source might be in Markdown format.
# 
# LangChain provides the `UnstructuredMarkdownLoader` to load content from Markdown files.
# 

# In[13]:


#get_ipython().system("wget 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/eMSP5vJjj9yOfAacLZRWsg/markdown-sample.md'")
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/eMSP5vJjj9yOfAacLZRWsg/markdown-sample.md"
output_file = "markdown-sample.md"

# Télécharger le fichier
response = requests.get(url)
if response.status_code == 200:
    with open(output_file, 'wb') as file:
        file.write(response.content)
else:
    print(f"Erreur : Impossible de télécharger le fichier (code {response.status_code})")


# In[14]:


markdown_path = "markdown-sample.md"
loader = UnstructuredMarkdownLoader(markdown_path)
print(loader)


# In[15]:


data = loader.load()


# In[16]:


print(data)


# ### Load from JSON files
# 
# 

# The JSONLoader uses a specified [jq schema](https://en.wikipedia.org/wiki/Jq_(programming_language)) to parse the JSON files. It uses the jq python package, which we've installed before.
# 

# In[17]:


#get_ipython().system("wget 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/hAmzVJeOUAMHzmhUHNdAUg/facebook-chat.json'")
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/hAmzVJeOUAMHzmhUHNdAUg/facebook-chat.json"
output_file = "facebook-chat.json"

# Télécharger le fichier
response = requests.get(url)
if response.status_code == 200:
    with open(output_file, 'wb') as file:
        file.write(response.content)
else:
    print(f"Erreur : Impossible de télécharger le fichier (code {response.status_code})")


# First, let's use `pprint` to take a look at the JSON file and its structure. 
# 

# In[18]:


file_path='facebook-chat.json'
data = json.loads(Path(file_path).read_text())


# In[19]:


pprint(data)


# We use `JSONLoader` to load data from the JSON file. However, JSON files can have various attribute-value pairs. If we want to load a specific attribute and its value, we need to set an appropriate `jq schema`.
# 
# So for example, if we want to load the `content` from the JSON file, we need to set `jq_schema='.messages[].content'`.
# 

# In[20]:


loader = JSONLoader(
    file_path=file_path,
    jq_schema='.messages[].content',
    text_content=False)

data = loader.load()


# In[21]:


pprint(data)


# ### Load from CSV files
# 

# CSV files are a common format for storing tabular data. The `CSVLoader` provides a convenient way to read and process this data.
# 

# In[22]:


#get_ipython().system("wget 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IygVG_j0M87BM4Z0zFsBMA/mlb-teams-2012.csv'")
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IygVG_j0M87BM4Z0zFsBMA/mlb-teams-2012.csv"
output_file = "mlb-teams-2012.csv"

# Télécharger le fichier
response = requests.get(url)
if response.status_code == 200:
    with open(output_file, 'wb') as file:
        file.write(response.content)
else:
    print(f"Erreur : Impossible de télécharger le fichier (code {response.status_code})")

# In[23]:


loader = CSVLoader(file_path='mlb-teams-2012.csv')
data = loader.load()


# In[24]:


print(data)


# When you load data from a CSV file, the loader typically creates a separate `Document` object for each row of data in the CSV.
# 

# #### UnstructuredCSVLoader
# 

# In contrast to `CSVLoader`, which treats each row as an individual document with headers defining the data, `UnstructuredCSVLoader` considers the entire CSV file as a single unstructured table element. This approach is beneficial when you want to analyze the data as a complete table rather than as separate entries.
# 

# In[25]:


loader = UnstructuredCSVLoader(
    file_path="mlb-teams-2012.csv", mode="elements"
)
data = loader.load()


# In[26]:


data[0].page_content


# In[27]:


print(data[0].metadata["text_as_html"])


# ### Load from URL/Website files
# 

# Usually we use `BeautifulSoup` package to load and parse a HTML or XML file. But it has some limitations.
# 
# The following code is using `BeautifulSoup` to parse a website. Let's see what limitation it has.
# 

# In[28]:


import requests
from bs4 import BeautifulSoup

url = 'https://www.ibm.com/topics/langchain'
response = requests.get(url)

soup = BeautifulSoup(response.content, 'html.parser')
print(soup.prettify())


# From the print output, we can see that `BeautifulSoup` not only load the web content, but also a lot of HTML tags and external links, which are not necessary if we just want to load the text content of the web.
# 

# So LangChain's `WebBaseLoader` can effectively address this limitation.
# 
# `WebBaseLoader` is designed to extract all text from HTML webpages and convert it into a document format suitable for further processing.
# 

# #### Load from single web page
# 

# In[29]:


loader = WebBaseLoader("https://www.ibm.com/topics/langchain")


# In[30]:


data = loader.load()


# In[31]:


print(data)


# #### Load from multiple web pages
# 

# You can load multiple webpages simultaneously by passing a list of URLs to the loader. This will return a list of documents corresponding to the order of the URLs provided.
# 

# In[32]:


loader = WebBaseLoader(["https://www.ibm.com/topics/langchain", "https://www.redhat.com/en/topics/ai/what-is-instructlab"])
data = loader.load()
print(data)


# ### Load from WORD files
# 

# `Docx2txtLoader` is utilized to convert Word documents into a document format suitable for further processing.
# 

# In[33]:


#get_ipython().system('wget "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/94hiHUNLZdb0bLMkrCh79g/file-sample.docx"')
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/94hiHUNLZdb0bLMkrCh79g/file-sample.docx"
output_file = "file-sample.docx"

# Télécharger le fichier
response = requests.get(url)
if response.status_code == 200:
    with open(output_file, 'wb') as file:
        file.write(response.content)
else:
    print(f"Erreur : Impossible de télécharger le fichier (code {response.status_code})")


# In[34]:


loader = Docx2txtLoader("file-sample.docx")


# In[35]:


data = loader.load()


# In[36]:


print(data)


# ### Load from Unstructured Files
# 

# Sometimes, we need to load content from various text sources and formats without writing a separate loader for each one. Additionally, when a new file format emerges, we want to save time by not having to write a new loader for it. `UnstructuredFileLoader` addresses this need by supporting the loading of multiple file types. Currently, `UnstructuredFileLoader` can handle text files, PowerPoints, HTML, PDFs, images, and more.
# 

# For example, we can load `.txt` file.
# 

# In[37]:


loader = UnstructuredFileLoader("new-Policies.txt")
data = loader.load()
print(data)


# We also can load `.md` file.
# 

# In[38]:


loader = UnstructuredFileLoader("markdown-sample.md")
data = loader.load()
print(data)


# #### Multiple files with different formats
# 

# We can even load a list of files with different formats.
# 

# In[39]:


files = ["markdown-sample.md", "new-Policies.txt"]


# In[40]:


loader = UnstructuredFileLoader(files)


# In[41]:


data = loader.load()


# In[42]:


print(data)


# # Exercises
# 

# ### Exercise 1 - Try to use other PDF loaders
# 
# There are many other PDF loaders in LangChain, for example, `PyPDFium2Loader`. Can you use this PDF loader to load the PDF and see the difference?
# 

# In[51]:


# Your code here
pdf_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/Q81D33CdRLK6LswuQrANQQ/instructlab.pdf"
#get_ipython().system('pip install pypdfium2')
from langchain_community.document_loaders import PyPDFium2Loader
loader = PyPDFium2Loader(pdf_url)
pages = loader.load()
#pages = loader.load_and_split()
print(pages[0])


# <details>
#     <summary>Click here for Solution</summary>
# 
# 
# ```python
# 
# !pip install pypdfium2
# 
# from langchain_community.document_loaders import PyPDFium2Loader
# 
# loader = PyPDFium2Loader(pdf_url)
# 
# data = loader.load()
# 
# ```
# 
# </details>
# 

# ### Exercise 2 - Load from Arxiv
# 

# Sometimes we have paper that we want to load from Arxiv, can you load this [paper](https://arxiv.org/abs/1605.08386) using `ArxivLoader`.
# 

# In[3]:


# Your code here
#get_ipython().system('pip install arxiv')
from langchain.document_loaders import ArxivLoader
docs = ArxivLoader(query="1605.08386", load_max_docs=2).load()
print(docs[0].page_content[:1000])


# <details>
#     <summary>Click here for Solution</summary>
#     
# ```python
# 
# !pip install arxiv
# 
# from langchain_community.document_loaders import ArxivLoader
# 
# docs = ArxivLoader(query="1605.08386", load_max_docs=2).load()
# 
# print(docs[0].page_content[:1000])
# 
# ```
# 
# </details>
# 

# ## Authors
# 

# [Kang Wang](https://www.linkedin.com/in/kangwang95/)
# 
# Kang Wang is a Data Scientist in IBM. He is also a PhD Candidate in the University of Waterloo.
# 

# ### Other Contributors
# 

# [Joseph Santarcangelo](https://www.linkedin.com/in/joseph-s-50398b136/)
# 
# Joseph has a Ph.D. in Electrical Engineering, his research focused on using machine learning, signal processing, and computer vision to determine how videos impact human cognition. Joseph has been working for IBM since he completed his PhD.
# 
# [Hailey Quach](https://author.skills.network/instructors/hailey_quach)
# 
# Hailey is a Data Scientist at IBM. She is also an undergraduate student at Concordia University, Montreal.
# 

# © Copyright IBM Corporation. All rights reserved.
# 
