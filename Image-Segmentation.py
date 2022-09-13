#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install opencv-python')


# In[3]:


import cv2


# In[4]:


img = cv2.imread("monarch.png")


# In[5]:


img


# In[6]:


import matplotlib.pyplot as plt


# In[7]:


plt.imshow(img)


# In[8]:


img.shape


# In[9]:


img.shape[0]


# In[10]:


img.shape[1]


# In[ ]:




