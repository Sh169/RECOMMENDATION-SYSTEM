#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from numpy import random, float, array
import numpy as np
import seaborn as sns


# In[41]:


books=pd.read_csv("rs3.csv")
books.head()


# In[42]:


#number of unique users in the dataset
len(books.userid.unique())


# In[43]:


len(books.booktitle.unique())


# In[51]:


ratings = pd.pivot_table(data=books, values='bookrating', index='userid', columns='booktitle')
ratings


# In[54]:


ratings.index = books.userid.unique()
ratings


# In[55]:


#Impute those NaNs with 0 values
ratings.fillna(0, inplace=True)


# In[56]:


ratings


# In[57]:


#Calculating Cosine Similarity between Users
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine, correlation


# In[58]:


user_sim = 1 - pairwise_distances(ratings.values,metric='cosine')


# In[59]:


user_sim


# In[60]:


#Store the results in a dataframe
user_sim_df = pd.DataFrame(user_sim)


# In[61]:


#Set the index and column names to user ids 
user_sim_df.index = books.userid.unique()
user_sim_df.columns = books.userid.unique()


# In[62]:


user_sim_df.iloc[0:5, 0:5]


# In[63]:


np.fill_diagonal(user_sim, 0)
user_sim_df.iloc[0:5, 0:5]


# In[64]:


#Most Similar Users
user_sim_df.idxmax(axis=1)[0:5]


# In[68]:


books[(books['userid']==276729) | (books['userid']==276847)]


# In[70]:


user_1=books[books['userid']==276729]


# In[71]:


user_2=books[books['userid']==276747]


# In[72]:


user_2.booktitle


# In[73]:


user_1.booktitle


# In[74]:


pd.merge(user_1,user_2,on='booktitle',how='outer')


# In[ ]:




