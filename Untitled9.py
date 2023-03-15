#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler


# In[3]:


df = pd.read_csv("/home/helio/Downloads/Mall_Customers.csv")


# In[10]:


df.info()


# In[11]:


df.head()


# In[ ]:





# In[4]:


X = df.iloc[:, -3:].values


# In[5]:


# Standardize the data
scaler = StandardScaler()
X_std = scaler.fit_transform(X)


# In[6]:


# Perform hierarchical clustering
model = AgglomerativeClustering(n_clusters=5)
model.fit(X_std)


# In[7]:


# Plot the output as a 3D scatter plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=model.labels_, cmap='rainbow')
ax.set_xlabel("Annual Income (k$)")
ax.set_ylabel("Spending Score (1-100)")
ax.set_zlabel("Age")
plt.show()


# In[ ]:




