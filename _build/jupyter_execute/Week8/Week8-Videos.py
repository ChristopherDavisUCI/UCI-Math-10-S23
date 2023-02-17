#!/usr/bin/env python
# coding: utf-8

# # Week 8 Videos

# ## Downloading the MNIST dataset of handwritten digits
# 
# <iframe width="560" height="315" src="https://www.youtube.com/embed/s9X9fdUmS0s" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
# 
# ![Handwritten 5](../images/mnist.png)
# 
# Source: [medium.com](https://medium.com/comet-ml/real-time-numbers-recognition-mnist-on-an-iphone-with-coreml-from-a-to-z-283161441f90)
# 
# * Find the dataset name on [openml.org](https://www.openml.org/) and load it into the notebook using the `fetch_openml` function from scikit-learn's `datasets` module.  (**Warning**.  I tried loading this dataset twice and ran out of memory, so only run the code once.)

# In[1]:


from sklearn.datasets import fetch_openml


# In[2]:


mnist = fetch_openml("mnist_784")


# In[3]:


type(mnist)


# In[4]:


dir(mnist)


# In[5]:


dfX = mnist.data


# In[6]:


type(dfX)


# In[7]:


dfX.shape


# In[8]:


dfX.iloc[4]


# In[9]:


arr = dfX.iloc[4].to_numpy()


# In[10]:


arr.shape


# In[11]:


arr2d = arr.reshape((28,28))


# In[12]:


arr2d.shape


# In[13]:


import matplotlib.pyplot as plt


# In[14]:


fig, ax = plt.subplots()
ax.imshow(arr2d)


# In[15]:


fig, ax = plt.subplots()
ax.imshow(arr2d, cmap='binary')


# In[16]:


fig, ax = plt.subplots()
ax.imshow(arr2d, cmap='binary_r')


# In[17]:


y = mnist.target


# In[18]:


type(y)


# In[19]:


y.iloc[4]


# 
# * Divide the data into a training set and a test set, using 90% of the samples for the training set.

# In[20]:


from sklearn.model_selection import train_test_split


# In[21]:


X_train, X_test, y_train, y_test = train_test_split(dfX, y, train_size=0.9)


# In[22]:


X_train.shape


# In[23]:


y_train.shape


# In[24]:


X_test.shape


# ## Bad idea: Using linear regression with MNIST
# 
# <iframe width="560" height="315" src="https://www.youtube.com/embed/ufc7TwFYxhY" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

# In[31]:


from sklearn.linear_model import LinearRegression


# In[58]:


reg = LinearRegression()


# In[59]:


reg.fit(X_train, y_train)


# In[60]:


reg.predict(X_train)


# In[61]:


z = reg.predict(X_train).round()


# In[62]:


z


# In[63]:


y_train


# In[64]:


y_train == 1


# In[65]:


y_train == '1'


# In[66]:


y2 = y_train.astype(int)


# In[67]:


y2 == 1


# In[68]:


y2 == z


# In[69]:


(y2 == z).mean()


# ## Using a decision tree with MNIST
# 
# <iframe width="560" height="315" src="https://www.youtube.com/embed/TOFFhHO_NNk" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

# In[70]:


from sklearn.tree import DecisionTreeClassifier


# In[71]:


clf = DecisionTreeClassifier()


# In[72]:


clf.fit(X_train, y_train)


# In[73]:


z = clf.predict(X_train)


# In[74]:


z


# In[76]:


(z == y_train).mean()


# In[77]:


z2 = clf.predict(X_test)


# In[78]:


(z2 == y_test).mean()


# In[ ]:





# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=d9849fdc-63d6-465a-a706-7821a3cb4d78' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>
