#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from __future__ import division


# In[2]:


# define working directory
os.chdir('/Users/kksivva/Downloads/datamining')


# In[3]:


sales = pd.read_csv('dataset.csv')


# In[4]:


# top 5 observations
sales = sales.rename(columns = {"gender":"Gender", "customerID":"CustomerID","tenure":"Tenure"}) 


# In[5]:


sales['Partner'] = sales['Partner'].map({
    "No":0,
    "Yes":1
}.get)
sales['Dependents'] = sales['Dependents'].map({
    "No":0,
    "Yes":1
}.get)
sales['PhoneService'] = sales['PhoneService'].map({
    "No":0,
    "Yes":1
}.get)
sales['PaperlessBilling'] = sales['PaperlessBilling'].map({
    "No":0,
    "Yes":1
}.get)
sales['Churn'] = sales['Churn'].map({
    "No":0,
    "Yes":1
}.get)
sales['Gender'] = sales['Gender'].map({
    "Male":0,
    "Female":1
}.get)


# In[6]:


sales.head()


# In[7]:


sales['MultipleLines'].unique()


# In[8]:


sales['MultipleLines'].replace({
    "No":0,
    "Yes":1,
    "No phone service":2
}, inplace=True)


# In[9]:


sales['InternetService'].unique()


# In[10]:


sales['InternetService'].replace({
    "No":0,
    "DSL":1,
    "Fiber optic":2
}, inplace=True)


# In[11]:


sales.head()


# In[12]:


sales['PaymentMethod'].unique()


# In[13]:


sales['PaymentMethod'].replace({
    "Electronic check":0,
    "Mailed check":1,
    "Bank transfer (automatic)":2,
    "Credit card (automatic)":3
}, inplace=True)


# In[14]:


sales['StreamingService'].unique()


# In[15]:


sales['StreamingService'].replace({
    "No":0,
    "Yes":1,
    "No internet service":2
}, inplace=True)


# In[16]:


sales.head()


# In[17]:


sales['Contract'].unique()


# In[18]:


sales['Contract'].replace({
    "Month-to-month":0,
    "One year":1,
    "Two year":2
}, inplace=True)


# In[19]:


sales.tail()


# In[20]:


#Normalize Monthly charges data
sales['MonthlyCharges'] = sales['MonthlyCharges'].apply(lambda v:(v - sales['MonthlyCharges'].min())/(sales['MonthlyCharges'].max() - sales['MonthlyCharges'].min()))


# In[21]:


sales.head()


# In[22]:


# Replace non-numeric values with 0
sales['TotalCharges'] = pd.to_numeric(sales['TotalCharges'], errors='coerce')


# In[23]:


sales.head()


# In[24]:


#Normalize Total charges data
sales['TotalCharges'] = sales['TotalCharges'].apply(lambda v:(v - sales['TotalCharges'].min())/(sales['TotalCharges'].max() - sales['TotalCharges'].min()))


# In[25]:


sales.head()


# In[26]:


sales['Tenure'].unique()


# In[27]:


sales.head()


# In[28]:


sales['Tenure'].min()


# In[29]:


sales.Tenure.max()


# In[30]:


sales['Tenure'] = sales['Tenure'].apply(lambda v:(v - sales['Tenure'].min())/(sales['Tenure'].max() - sales['Tenure'].min()))


# In[31]:


sales.head()


# In[32]:


sales.describe()


# In[33]:


sales.to_csv("preprocessed_dataset.csv", index=False)


# In[34]:


new_df = pd.read_csv("preprocessed_dataset.csv")


# In[35]:


new_df.head()


# In[36]:


new_df.corr()


# In[37]:


get_ipython().magic(u'matplotlib inline')


# In[38]:


plt.matshow(new_df.corr(),cmap="summer")
plt.colorbar()

plt.xticks(list(range(len(new_df.columns))), new_df.columns,rotation='vertical')
plt.yticks(list(range(len(new_df.columns))), new_df.columns,rotation='horizontal')

plt.show()


# In[39]:


new_df.corr()["Churn"].sort_values(ascending=False)


# In[40]:


new_df.plot.box()
plt.xticks(list(range(len(new_df.columns))), new_df.columns,rotation='vertical')



# In[43]:


#covariance matrix
pd.plotting.scatter_matrix(new_df, alpha=0.2, figsize=(18,18), diagonal='kde')
plt.show()


# In[45]:


new_df.cov()


# In[48]:


new_df.plot.kde(subplots=True,figsize=(18,18))


# In[ ]:




