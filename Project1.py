#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np 
import seaborn as sns


# In[5]:


data=pd.read_csv("C:/Users/Faizaan/Downloads/Methane/methane_hist_emissions.csv")


# In[18]:


data.dtypes
data['2018']=data['2018'].astype(int)
data['2017']=data['2017'].astype(int)


# In[20]:


data.dtypes


# In[21]:


data.describe()


# In[10]:


data.isnull().sum()


# In[22]:


data.head()


# In[50]:


#g = sns.relplot(x='2018', y='Sector', hue='Gas',data=data)
#g.set_axis_labels("2018","Sector")

from matplotlib import pyplot as plt
plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

sns.set_style("darkgrid")
tips = sns.load_dataset("tips")

ax = sns.boxplot(x="2018", y="2017",hue='Sector', data=data)
ax.set(xlabel=None)

plt.show()


# In[24]:


data.head()


# In[33]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[34]:


train = data.drop(['Unit','Gas','Country','Sector','1990'], axis=1)
test = data['2018']


# In[35]:


X_train,X_test,y_train,y_test = train_test_split(train,test,test_size=0.3, random_state=2)


# In[36]:


regr = LinearRegression()


# In[37]:


regr.fit(X_train,y_train)


# In[38]:


pred =regr.predict(X_test)
data.head()


# In[39]:


regr.score(X_test,y_test)


# In[ ]:




