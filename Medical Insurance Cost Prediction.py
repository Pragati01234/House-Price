#!/usr/bin/env python
# coding: utf-8

# # Importing the Dependencies

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


# # Data Collection & Analysis

# In[2]:


# loading the data from csv file to a Pandas DataFrame
insurance_dataset = pd.read_csv('D:\panda_files/insurance.csv')


# In[3]:


# first 5 rows of the dataframe
insurance_dataset.head()


# In[4]:


# number of rows and columns
insurance_dataset.shape


# In[5]:


# getting some informations about the dataset
insurance_dataset.info()


# # Categorical Features:
# 
# * Sex
# * Smoker
# * Region

# In[6]:


# checking for missing values
insurance_dataset.isnull().sum()


# # Data Analysis

# In[7]:


# statistical Measures of the dataset
insurance_dataset.describe()


# In[8]:


# distribution of age value
sns.set()
plt.figure(figsize=(6,6))
sns.distplot(insurance_dataset['age'])
plt.title('Age Distribution')
plt.show()


# In[9]:


# Gender column
plt.figure(figsize=(6,6))
sns.countplot(x='sex', data=insurance_dataset)
plt.title('Sex Distribution')
plt.show()


# In[10]:


insurance_dataset['sex'].value_counts()


# In[11]:


# bmi distribution
plt.figure(figsize=(6,6))
sns.distplot(insurance_dataset['bmi'])
plt.title('BMI Distribution')
plt.show()


# # Normal BMI Range --> 18.5 to 24.9

# In[12]:


# children column
plt.figure(figsize=(6,6))
sns.countplot(x='children', data=insurance_dataset)
plt.title('Children')
plt.show()


# In[13]:


insurance_dataset['children'].value_counts()


# In[14]:


# smoker column
plt.figure(figsize=(6,6))
sns.countplot(x='smoker', data=insurance_dataset)
plt.title('smoker')
plt.show()


# In[15]:


insurance_dataset['smoker'].value_counts()


# In[16]:


# region column
plt.figure(figsize=(6,6))
sns.countplot(x='region', data=insurance_dataset)
plt.title('region')
plt.show()


# In[17]:


insurance_dataset['region'].value_counts()


# In[18]:


# distribution of charges value
plt.figure(figsize=(6,6))
sns.distplot(insurance_dataset['charges'])
plt.title('Charges Distribution')
plt.show()


# # Data Pre-Processing
# 
# Encoding the categorical features

# In[19]:


# encoding sex column
insurance_dataset.replace({'sex':{'male':0,'female':1}}, inplace=True)

3 # encoding 'smoker' column
insurance_dataset.replace({'smoker':{'yes':0,'no':1}}, inplace=True)

# encoding 'region' column
insurance_dataset.replace({'region':{'southeast':0,'southwest':1,'northeast':2,'northwest':3}}, inplace=True)


# # Splitting the Features and Target

# In[20]:


X = insurance_dataset.drop(columns='charges', axis=1)
Y = insurance_dataset['charges']


# In[21]:


print(X)


# In[22]:


print(Y)


# In[23]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)


# In[24]:


print(X.shape, X_train.shape, X_test.shape)


# # Model Training

# Linear Regression

# In[25]:


# loading the Linear Regression model
regressor = LinearRegression()


# In[26]:


regressor.fit(X_train, Y_train)


# # Model Evaluation

# In[27]:


# prediction on training data
training_data_prediction =regressor.predict(X_train)


# In[28]:


# R squared value
r2_train = metrics.r2_score(Y_train, training_data_prediction)
print('R squared vale : ', r2_train)


# In[29]:


# prediction on test data
test_data_prediction =regressor.predict(X_test)


# In[30]:


# R squared value
r2_test = metrics.r2_score(Y_test, test_data_prediction)
print('R squared vale : ', r2_test)


# # Building a Predictive System

# In[31]:


input_data = (31,1,25.74,0,1,0)

# changing input_data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = regressor.predict(input_data_reshaped)
print(prediction)

print('The insurance cost is USD ', prediction[0])


# In[32]:


input_data = (211,3,34,0,1,0)

# changing input_data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = regressor.predict(input_data_reshaped)
print(prediction)

print('The insurance cost is USD ', prediction[0])


# In[ ]:




