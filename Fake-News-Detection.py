#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import re
import string


# In[2]:


df_fake = pd.read_csv("D:\panda\Fake.csv")
df_true = pd.read_csv("D:\panda\True.csv")


# In[3]:


df_fake.head(5)


# In[4]:


df_true.head(5)


# In[5]:


df_fake["class"] = 0
df_true["class"] = 1


# In[6]:


df_fake.shape, df_true.shape


# In[7]:


df_fake_manual_testing = df_fake.tail(10)
for i in range(23480,23470,-1):
    df_fake.drop([i], axis = 0, inplace = True)
df_true_manual_testing = df_true.tail(10)
for i in range(21416,21406,-1):
    df_true.drop([i], axis = 0, inplace = True)


# In[8]:


df_fake.shape, df_true.shape


# In[9]:


df_fake_manual_testing["class"] = 0
df_true_manual_testing["class"] = 1


# In[10]:


df_fake_manual_testing.head(10)


# In[11]:


df_true_manual_testing.head(10)


# In[12]:


df_manual_testing = pd.concat([df_fake_manual_testing,df_true_manual_testing], axis = 0)
df_manual_testing.to_csv("manual_testing.csv")


# In[13]:


df_marge = pd.concat([df_fake, df_true], axis =0 )
df_marge.head(10)


# In[14]:


df_marge.columns


# In[15]:


df = df_marge.drop(["title", "subject","date"], axis = 1)


# In[16]:


df.isnull().sum()


# In[17]:


df = df.sample(frac = 1)
df.head()


# In[18]:


df.reset_index(inplace = True)
df.drop(["index"], axis = 1, inplace = True)


# In[19]:


df.columns


# In[20]:


df.head()


# In[21]:


def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) 
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)    
    return text


# In[22]:


df["text"] = df["text"].apply(wordopt)


# In[23]:


x = df["text"]
y = df["class"]


# In[24]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)


# In[25]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[26]:


vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)


# In[27]:


from sklearn.linear_model import LogisticRegression


# In[28]:


LR = LogisticRegression()
LR.fit(xv_train,y_train)


# In[30]:


pred_lr=LR.predict(xv_test)


# In[31]:


LR.score(xv_test, y_test)


# In[32]:


print(classification_report(y_test, pred_lr))


# In[33]:


from sklearn.tree import DecisionTreeClassifier


# In[34]:


DT = DecisionTreeClassifier()
DT.fit(xv_train, y_train)


# In[35]:


pred_dt = DT.predict(xv_test)


# In[36]:


DT.score(xv_test, y_test)


# In[37]:


print(classification_report(y_test, pred_dt))


# In[38]:


from sklearn.ensemble import GradientBoostingClassifier


# In[39]:


GBC = GradientBoostingClassifier(random_state=0)
GBC.fit(xv_train, y_train)


# In[40]:


pred_gbc = GBC.predict(xv_test)


# In[41]:


GBC.score(xv_test, y_test)


# In[42]:


print(classification_report(y_test, pred_gbc))


# In[46]:


from sklearn.ensemble import RandomForestClassifier


# In[47]:


RFC = RandomForestClassifier(random_state=0)
RFC.fit(xv_train, y_train)


# In[48]:


pred_rfc = RFC.predict(xv_test)


# In[49]:


RFC.score(xv_test, y_test)


# In[50]:


print(classification_report(y_test, pred_rfc))


# In[51]:


def output_lable(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not A Fake News"
    
def manual_testing(news):
    testing_news = {"text":[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt) 
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_LR = LR.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)
    pred_GBC = GBC.predict(new_xv_test)
    pred_RFC = RFC.predict(new_xv_test)

    return print("\n\nLR Prediction: {} \nDT Prediction: {} \nGBC Prediction: {} \nRFC Prediction: {}".format(output_lable(pred_LR[0]), 
                                                                                                              output_lable(pred_DT[0]), 
                                                                                                              output_lable(pred_GBC[0]), 
                                                                                                              output_lable(pred_RFC[0])))


# In[ ]:


news = str(input("Paul Craig RobertsIn the last years of the 20th century fraud entered US foreign policy in a new way"))
manual_testing(news)


# In[ ]:




