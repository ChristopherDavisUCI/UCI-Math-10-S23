#!/usr/bin/env python
# coding: utf-8

# # NFL offense performance and analysis
# 
# Author: Shengkai Yang
# 
# Course Project, UC Irvine, Math 10, F22

# ## Introduction
# 
# In this project, I want to use machine learning to analyze how some specific data which affect NFL football games. Also, I want to know how these data affect teams performance and predict them.

# ## Main part of project
# 
# You can either have all one section or divide into multiple sections.  To make new sections, use `##` in a markdown cell.  Double-click this cell for an example of using `##`

# In[1]:


import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns


# In[2]:


w4 = pd.read_csv("nfloffenseweek4.csv")


# In[3]:


w4


# First of all, I need to clean the data, based on the original data, I have the data of rushing touchdowns and passing touchdowns, I want to collect all of them together to make sure how many touchdowns they made.
# 

# In[4]:


w4["touchdowns"] = w4["passing_touchdowns"]+w4["rushing_touchdowns"]
w4


# In[5]:


w4["lost"] = w4["turnovers_lost"]+w4["fumbles_lost"]
w4


# ## Using Altair chart to show the relationship between their offense performance and the rank
# 

# In[6]:


c1 = alt.Chart(w4).mark_circle().encode(
    x="yards_per_play",
    y="points_scored",
    color =alt.Color("rank", scale=alt.Scale(scheme="goldgreen")),
    tooltip =["team", "yards_per_play", "points_scored"]
)


# In[7]:


c1


# In[8]:


w4.columns


# from above altair chart, we can tell lighter yellow means higher rank, darker green means lower rank, in the middle part. Since Green and Yello is totally different, so we can directly see the relationship between rank and offense performance. Miami Dolphins gets top yards_per_play in the league, but the rank is low.

# ## Using DecisionTree to classify to predict

# I want to find the relationship between yards per play and teams' touchdowns, so I use decision tree to predict them.

# In[9]:


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from pandas.api.types import is_numeric_dtype


# In[10]:


w4_1 = w4[["points_scored","yards_per_play","offensive_plays","passes_completed","passes_attempted","touchdowns"]]
w4_1


# In[11]:


num_cols = [c for c in w4_1.columns if is_numeric_dtype(w4[c])]
num_cols


# In[12]:


features = [c for c in w4_1 if c != "touchdowns"]
features


# In[13]:


x = w4_1[features]
y = w4_1["touchdowns"]


# In[14]:


X_train, X_test, y_train, y_test = train_test_split(w4_1, w4_1["touchdowns"], test_size=0.2, random_state=0)


# In[15]:


clf = DecisionTreeClassifier(max_depth=6)


# In[16]:


X_test
y_test


# In[17]:


X_train
y_train


# In[18]:


clf.fit(X_train, y_train)


# In[19]:


clf.score(X_train, y_train)


# In[20]:


clf.score(X_test, y_test)


# In[21]:


from sklearn.tree import plot_tree
import matplotlib.pyplot as plt


# In[22]:


fig = plt.figure(figsize=(9,15))
plot_tree(
        clf,
        feature_names=clf.feature_names_in_,
        filled=True
        
    );


# Based on above information, I choose max_depth is 6, and the train set value is 0.96 and the test value is round to 0.86, so this is not overfitting in this model. Also in the above Decission chart, we can see the different situation's result. Like touchdowns less or equal than 9.5,etc.

# ## Using K-Neighbors to predict

# This is the extra part of Math10, because this classifier is supervised and make classificartions, predictions about individual data point in a group, so I think it is good for analyze NFL datas. Even though peopla call it "lazy".

# In[23]:


from sklearn.neighbors import KNeighborsClassifier


# In[27]:


scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)


# In[25]:


X = w4[["rushing_attempts","passes_attempted"]]
y = w4["points_scored"]


# In[28]:


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, train_size=0.6, random_state=4)


# In[30]:


clf2 = KNeighborsClassifier()
clf2.fit(X_train, y_train)


# In[47]:


w4_1["pred"] = clf2.predict(X_scaled)
clf2.fit(X_train, y_train)


# In[52]:


w4_1


# In[60]:


c3 = alt.Chart(w4_1).mark_circle().encode(
    x="passes_completed",
    y="passes_attempted",
    color=alt.Color("pred", title="rank"),
    tooltip = ('passes_attempted','touchdowns','points_scored','yards_per_play','passes_completed')
).properties(
    title="Passing",
    width=400,
    height=400,
)
c3


# Using KNeighborsCalssifier and altair chart to show the performance of passing in each team, except two teams, which is almost 50% passing complete on the left side, and another one 193 passes attempts and 111 completed, high passes, high rate of success. other team has the similar rate of passing success. We can see there is a line in the graph.

# ## Summary
# In the final project, I use altair chart, Decission Tree and KNeighbor classifier to analyze the NFL teams offense performance. Also, using Decission Tree to make sure whtether it is overfitting or not which is very important in machine learning. 

# ## References
# 
# Your code above should include references.  Here is some additional space for references.

# * What is the source of your dataset(s)?
# https://www.kaggle.com/datasets/kendallgillies/nflstatistics

# * List any other references that you found helpful.
# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

# ## Submission
# 
# Using the Share button at the top right, **enable Comment privileges** for anyone with a link to the project. Then submit that link on Canvas.

# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=f8a7d75c-60f8-4572-9513-6dbb4ae3d40d' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>
