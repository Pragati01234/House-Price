#!/usr/bin/env python
# coding: utf-8

# # A brief about Support Vector Machine Model
# The algorithm that we shall be using for this purpose, is the Support Vector Machine. 
# Support Vector Machine,(SVM), falls under the “supervised machine learning algorithms” category.
# It can be used for classification, as well as for regression.
# In this model, we plot each data item as a unique point in an n-dimension,(where n is actually, the number of features that we have), with the value of each of the features being the value of that particular coordinate. 
# Then, we perform the process of classification by finding the hyper-plane that differentiates the two classes.

# # Importing Dependencies : 
# Let us first import all the modules and libraries that we are going to use in the future while making the project. The dependencies that we will be using are :
# 
# numpy, pandas, seaborn, and ScikitLearn.

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


# # Data Collection and Processing

# In[2]:


# loading the dataset to pandas DataFrame
loan_dataset = pd.read_csv('D:\panda_files\loan.csv')


# In[3]:


type(loan_dataset)


# In[4]:


# printing the first 5 rows of the dataframe
loan_dataset.head()


# In[5]:


# number of rows and columns
loan_dataset.shape


# In[6]:


# statistical measures
loan_dataset.describe()


# In[7]:


# number of missing values in each column
loan_dataset.isnull().sum()


# In[8]:


# dropping the missing values
loan_dataset = loan_dataset.dropna()


# In[9]:


# number of missing values in each column
loan_dataset.isnull().sum()


# In[10]:


# label encoding
loan_dataset.replace({"Loan_Status":{'N':0,'Y':1}},inplace=True)


# In[11]:


# printing the first 5 rows of the dataframe
loan_dataset.head()


# In[12]:


# Dependent column values
loan_dataset['Dependents'].value_counts()


# In[13]:


# replacing the value of 3+ to 4
loan_dataset = loan_dataset.replace(to_replace='3+', value=4)


# # Data Visualization

# # education & Loan Status
# sns.countplot(x='Education',hue='Loan_Status',data=loan_dataset)

# In[15]:


# education & Loan Status
sns.countplot(x='Education',hue='Loan_Status',data=loan_dataset)


# In[16]:


# marital status & Loan Status
sns.countplot(x='Married',hue='Loan_Status',data=loan_dataset)


# In[17]:


# convert categorical columns to numerical values
loan_dataset.replace({'Married':{'No':0,'Yes':1},'Gender':{'Male':1,'Female':0},'Self_Employed':{'No':0,'Yes':1},
                      'Property_Area':{'Rural':0,'Semiurban':1,'Urban':2},'Education':{'Graduate':1,'Not Graduate':0}},inplace=True)


# In[18]:


loan_dataset.head()


# In[19]:


# separating the data and label
X = loan_dataset.drop(columns=['Loan_ID','Loan_Status'],axis=1)
Y = loan_dataset['Loan_Status']


# In[20]:


print(X)
print(Y)


# # Splitting X and Y into Training and Testing Variables
# Now, we will be splitting the data into four variables, viz., X_train, Y_train, X_test, Y_test.

# In[21]:


X_train, X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.1,stratify=Y,random_state=2)


# In[22]:


print(X.shape, X_train.shape, X_test.shape)


# # Training the model:
# 
# Support Vector Machine Model

# In[23]:


classifier = svm.SVC(kernel='linear')


# In[24]:


#training the support Vector Macine model
classifier.fit(X_train,Y_train)


# # Model Evaluation

# In[25]:


# accuracy score on training data
X_train_prediction = classifier.predict(X_train)
training_data_accuray = accuracy_score(X_train_prediction,Y_train)


# In[26]:


print('Accuracy on training data : ', training_data_accuray)


# In[27]:


# accuracy score on training data
X_test_prediction = classifier.predict(X_test)
test_data_accuray = accuracy_score(X_test_prediction,Y_test)


# In[28]:


print('Accuracy on test data : ', test_data_accuray)


# # Making a predictive system
# 
# 

# In[ ]:





# In[ ]:




