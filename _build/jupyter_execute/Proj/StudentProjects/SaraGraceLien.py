#!/usr/bin/env python
# coding: utf-8

# # Applying Machine Learning techniques to the Fashion MNIST data set
# 
# Author: Sara-Grace Lien
# 
# Email: sjlien@uci.edu or saragracelien@gmail.com 
# 
# Course Project, UC Irvine, Math 10, F22

# ## Introduction
# 
# For this project, I will be analysing the FashionMNIST data set and using machine learning models to classify the item to its respective label. I will analyse the results using plotting tools like Altair.

# ## Exploring and loading the data
# 
# In this portion of the project, I will use Pandas to explore the data.

# In[68]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# In[2]:


df_train = pd.read_csv('fashion-mnist_train.csv')
df_test = pd.read_csv('fashion-mnist_test.csv')


# In[3]:


df_test


# In[67]:


df_train


# ### Checking or any empty values using .isna()

# Because the data i downloaded came in 2 different datasets, I will be analyzing both just in case. 

# In[4]:


# looking for any NA 
df_train.isna().any().sum()
#we use .sum() to find how many NA values there are. 


# In[5]:


df_test.isna().any(axis=0).sum()


# We can conclude that there are no empty values in the data set.

# In[6]:


#type of data 
type(df_test)


# In[7]:


type(df_train)


# In[8]:


df_test.columns


# In[9]:


df_train.columns


# In[10]:


df_train.dtypes


# In[11]:


df_test.dtypes


# In[12]:


df_train.shape


# In[13]:


df_test.shape


# In[14]:


df_train['label']


# In[15]:


print(df_train.groupby(['label']).size())


# In[16]:


df_test['label']


# In[17]:


print(df_test.groupby(['label']).size())


# **Conclusion**
# I wanted to know how many images of each clothing item there are in the data set, so I used groupby to find the values. For both the test and train datasets, each piece of clothing has the same number of images. From looking at the dataframe, I found that the first column is the label, so later on, I would have to use that as the y value and the rest of the columns is the value of each pixel in the 28 by 28 image of each clothing.

# There are no obvious differences in the data other than its size, which we need as training set. 

# **Loading the data**
# Since the data is already split into test and train sets, I will not be using the train_test_split function from sklearn. What we are doing below is normalizing the data so it is similar across both training and testing data. We started from 1 for images because index 0 is the label. so x_train is the training data for the images and y_train is the respective label for the images. This is the same for the test data. Instead of having the pixel data from 0 to 255, we want to be 0 to 1 so we normalize it by dividing the data by 255. This makes working with the data more cohesive.

# In[ ]:


train = np.array(df_train, dtype='float32')
test = np.array(df_test, dtype='float32')


x_train = train[:, 1:] / 255
y_train = train[:, 0]

x_test = test[:, 1:] / 255
y_test = test[:, 0]


# In[62]:


x_train


# In[64]:


y_train


# Here we are using matplotlib to display an example of what the an image looks like. I used a random index for this to showcase a random item.

# In[84]:


image = x_train[9].reshape((28, 28)) # we can use any index

plt.imshow(image)
plt.show()


# Reference: 
# https://www.youtube.com/watch?v=N3oMKS1AfVI&ab_channel=MarkJay 
# https://www.kaggle.com/code/kutubkapadia/fashion-mnist-with-tensorflow 

# ## Fitting a random forest classifier
# 
# Reference: https://deepnote.com/workspace/week-1-worksheet-72ab4bc3-fed1-4bd4-abed-ff7237e046f1/project/Worksheet-15-Duplicate-1f29eb40-55f3-4d58-bd27-33664f91cedc/%2FWorksheet15.ipynb 
# 

# In[29]:


from sklearn.ensemble import RandomForestClassifier


# In[30]:


rfc = RandomForestClassifier(n_estimators= 150, max_depth= 150, max_leaf_nodes= 150, random_state=1234)


# In[31]:


rfc.fit(x_train,y_train)


# In[32]:


rfc_pred = rfc.predict(x_train)


# In[33]:


(rfc_pred == y_train).mean()


# In[34]:


rfc_pred2 = rfc.predict(x_test)


# In[35]:


(rfc_pred2 == y_test).mean()


# ### Conclusions from RandomForestClassifier

# The RandomForestClassifier is a model that harnesses the power of multiple Decision Trees. It does not rely on the feature importance of a single decision tree and since the FashionMNIST dataset is a big dataset, it is better to have its randomized feature selection. This is why I used this instead of a DecisionTreeClassifier. The train and test set are really close which means the model is doing a good job at classifying the images. This is how we know its learning because the accuracy for the test set is close to the train set.
# 
# **Why classification model?**
# 
# Although the data labels are numbers, they correspond to respective classes. There are no independent and dependent variables in the data, so a regression model would not make sense to use. 
# 
# Reference: 
# https://www.analyticsvidhya.com/blog/2020/05/decision-tree-vs-random-forest-algorithm/#h2_6 
# https://www.simplilearn.com/regression-vs-classification-in-machine-learning-article 

# ## Using plotting tools to analyse the data

# ### Making a Confusion Matrix using altair

# In[36]:


import pandas as pd
df = pd.DataFrame(y_test, columns=['Clothing'])
df["Pred"] = rfc.predict(x_test)
df


# In[61]:


# Confusion Matrix

import altair as alt
alt.data_transformers.enable('default', max_rows=15000)

c = alt.Chart(df).mark_rect().encode(
    x="Clothing:N",
    y="Pred:N",
    color = alt.Color("count()",scale=alt.Scale(scheme="plasma"))
    
)

c_text = alt.Chart(df).mark_text(color="white").encode(
    x="Clothing:N",
    y="Pred:N",
    text="count()"
    
)

(c+c_text).properties(
    height=400,
    width=400
)


# **What did we learn from this chart?** 
# 
# Let's refer to the labels from the FashionMNIST data set page on Kaggle found here: https://www.kaggle.com/datasets/zalando-research/fashionmnist 
# 
# Each training and test example is assigned to one of the following labels:
# 
# 0 T-shirt/top
# 1 Trouser
# 2 Pullover
# 3 Dress
# 4 Coat
# 5 Sandal
# 6 Shirt
# 7 Sneaker
# 8 Bag
# 9 Ankle boot
# 
# I'm going to analyse T-shirt/tops, coats,and shirts for this porton of the project as based on the graph, it seems like they get misclassified the most

# **T-shirt/Top**

# In[38]:


df.loc[df["Pred"] == 0, "Clothing"].value_counts(sort=False)


# T-shirts/Tops are most likely to be misclassified as a shirts. This makes sense because they are quite similar. The labelling of the data makes this confusing as T-shirt, tops and shirts are used interchangeably for most people (at least for me).
# 
# **Coat**

# In[39]:


df.loc[df["Pred"] == 4, "Clothing"].value_counts(sort=False)


# Coats are most likely to be misclassified as a shirt and a pullover. This makes sense as the shape of a shirt and pullover are similar to Coats.
# 
# **Shirts**
# 
# We can infer from the previous analysis that a lot of other clothing gets misclassified as a shirt. But let's take a look at what a shirt is most likely to get misclassified as. 

# In[40]:


df.loc[df["Pred"] == 6, "Clothing"].value_counts(sort=False)


# This is interesting because most items have counts from in between 800 to 1000 for the correct classification but shirts only has around 400 to 500 counts of the right classification(at least for the ones I ran). This is probably because shirts are misclassified so often. I think this is due to how vague the labels are. It might be better to split shirts into sub-categories like button-up shirt or blouse. 

# ## K-Nearest Neighbors Classifier
# This machine learning model calculate distances for each image to find its closest neighbours before deciding what label it belongs to. The goal of this protion is to find how many neighbors we need to produce the most accurate classifications without over or underfitting the data.
# Reference: https://christopherdavisuci.github.io/UCI-Math-10-W22/Week6/Week6-Wednesday.html 

# In[41]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error


# In[42]:


clf2 = KNeighborsClassifier(n_neighbors=10)


# In[43]:


clf2.fit(x_train,y_train)


# In[44]:


clf2.predict(x_test)


# In[45]:


x_test.shape


# In[46]:


x_train.shape


# The dataset is too large. We need to cut down the dataset so it doesnt take too long to run.

# In[47]:


x_train2 = x_train[0:1000]
x_test2 = x_test[0:1000]


# In[48]:


y_train2 = y_train[0:1000]
y_test2 = y_test[0:1000]


# In[49]:


mean_absolute_error(clf2.predict(x_test2), y_test2)


# In[50]:


mean_absolute_error(clf2.predict(x_train2), y_train2)


# Now that we have found the difference in accuracy for test and train data, we are next trying to find for what n_neighbour values will produce the best results. With this, we will use a test error curve. We will create a functioin that completes the above for a range of values n_neighbours will be. 

# In[51]:


def get_scores(k):
    clf2 = KNeighborsClassifier(n_neighbors=k)
    clf2.fit(x_train2, y_train2)
    train_error = mean_absolute_error(clf2.predict(x_train2), y_train2)
    test_error = mean_absolute_error(clf2.predict(x_test2), y_test2)
    return (train_error, test_error)


# In[52]:


df_scores = pd.DataFrame({"k":range(1,100),"train_error":np.nan,"test_error":np.nan})


# In[53]:


df_scores


# In[54]:


get_ipython().run_cell_magic('timeit', '', 'for i in df_scores.index:\n    df_scores.loc[i,["train_error","test_error"]] = get_scores(df_scores.loc[i,"k"])\n')


# In[55]:


df_scores


# Inverse K for more flexibility.

# In[56]:


df_scores["kinv"] = 1/df_scores.k


# In[57]:


df_scores


# In[90]:


ctrain_point = alt.Chart(df_scores).mark_circle().encode(
    x = "kinv",
    y = "train_error",
    tooltip = ['kinv', 'test_error','train_error', 'k']
).interactive()
ctest_point = alt.Chart(df_scores).mark_circle(color="orange").encode(
    x = "kinv",
    y = "test_error",
    tooltip= ['kinv', 'test_error','train_error', 'k']
).interactive()
scatter = ctrain_point+ctest_point


# In[92]:


ctrain_line = alt.Chart(df_scores).mark_line().encode(
    x = "kinv",
    y = "train_error",
    tooltip = ['kinv', 'test_error','train_error', 'k']
).interactive()
ctest_line = alt.Chart(df_scores).mark_line(color="orange").encode(
    x = "kinv",
    y = "test_error",
    tooltip= ['kinv', 'test_error','train_error', 'k']
).interactive()
line = ctrain_line+ctest_line


# In[93]:


scatter+line


# **What do we learn from these charts?**
# 
# With high n_neighbours values, k, the model is underfitting, whereas with lower n_neighbours values,k, the model is overfitting.  Because high n_neighbours values leads to more restrictions on the model, making it harder for it to classify the images. As for lower n_neighbour values, it has less restrictions, so the algorithm fits exactly to the training data, making it harder for it to predict future observations reliably.
# 
# 

# ## Summary
# 
# I used Pandas to better understand the data and plotting tools to analyse the predictions in the model. Based on the results, I drew conclusions about what the results are trying to tell us about the model and found the best restrictions for each model. From this project, I applied previous knoelwdge of Random Forest Classifier to classify the model and introduced the K-Nearest Neighbors Classifier for the 'Extra' portion of the project.

# ## References
# 
# Your code above should include references.  Here is some additional space for references.

# * What is the source of your dataset(s)? 
# 
# https://www.kaggle.com/datasets/zalando-research/fashionmnist 

# * List any other references that you found helpful.
# 
# https://www.datacamp.com/tutorial/k-nearest-neighbor-classification-scikit-learn 
# https://www.bmc.com/blogs/data-normalization/
# https://www.w3schools.com/python/numpy/numpy_array_slicing.asp 

# ## Submission
# 
# Using the Share button at the top right, **enable Comment privileges** for anyone with a link to the project. Then submit that link on Canvas.

# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=16214c36-ba59-4381-9d28-fc9cd0eb50c3' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>
