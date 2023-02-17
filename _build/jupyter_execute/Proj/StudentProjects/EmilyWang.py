#!/usr/bin/env python
# coding: utf-8

# # Predicting Heart Disease
# 
# Author: Emily Wang
# 
# Course Project, UC Irvine, Math 10, F22

# ## Introduction
# 
# Cardiovascular disease is one of the leading causes of death in the world. So it is important to know what risk factors play a role in determining heart disease, which will lead to helping to find a cure or solution to prevent heart disease. In my project, I am trying to predict whether or not a person has a heart disease based on certain risk factors. In addition, I will be looking at which factors have a greater chance of developing heart disease and which ones have no correlation.

# ## Main portion of the project

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import altair as alt
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn import tree


# ### Reading Data

# In[2]:


df = pd.read_csv("heart_data.csv")
df


# In[3]:


df.shape


# In[4]:


print(f"There are {df.shape[0]} rows and {df.shape[1]} columns in this dataset.")


# ### Cleaning Data

# In[5]:


# Checking to see if there are any missing values in the dataset.
df.isna().sum()


# The dataset doesn't have any missing data.

# In[6]:


df.dtypes


# In[7]:


df.columns


# In[8]:


# Sorting the dataset columns based on if they are numeric types or not.
num_cols = [cols for cols in df.select_dtypes(include=np.number)]
num_cols


# In[9]:


# creating a list of nonnumerical columns in the data set
categorical_cols = [cols for cols in df.select_dtypes(include="object")]
categorical_cols


# In[10]:


df["HD"] = df["HeartDisease"].map({0:"False", 1:"True"})


# To use the HeartDisease column as the output value and y variable for classification in the decision tree, the datatype needs to be changed to object since, in the original data set, the value was an integer datatype.

# In[11]:


df


# ### Data Analysis

# In[12]:


df.sample(10)


# In[13]:


df.info()


# In[14]:


df["HeartDisease"].value_counts()


# In this heart failure prediction dataset, 508/918, or about 55% of the people, had a heart condition. In the dataset's heart condition column, the value 1 means the person has a heart disease, and 0 means no heart disease was detected.

# In[15]:


dataplot=sns.heatmap(df.corr(), annot=True)


# - Based on the correlation matrix, there isn't a strong correlation between the different cardiovascular risk factors and the target values. 
# - The risk factor Oldpeak (numeric value for depression measure) has the highest positive correlation with heart disease, which means that if depression measure increases, then heart disease likelihood also increases and vice versa for the opposite direction.
# - Risk factor MaxHR (Maximum Heart Rate) has the highest negative correlation with HeartDisease, which means that as one variable increases, the other variable decreases. So if MaxHR increases, the likelihood of heart disease decreases.  
# - There is a surprising finding in this data: the correlation between cholesterol and heart disease is negative. 

# ### Train_Test_split

# In[16]:


num_cols.remove("HeartDisease")


# In[17]:


X = df[num_cols]
y = df["HD"]


# In[18]:


X.columns


# In[19]:


scaler = StandardScaler()


# In[20]:


X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = 0.8, random_state = 1)


# In[21]:


X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[22]:


X_train.shape


# In[23]:


y_train.shape


# ### Decision Tree Classifier

# In[24]:


X1 = df[num_cols]
y1 = df["HD"]


# In[25]:


X1_train, X1_test, y1_train, y1_test = train_test_split(X1,y1, train_size = 0.8, random_state = 1)


# In[26]:


clf = DecisionTreeClassifier(max_leaf_nodes=7, max_depth=10, random_state=1)


# In[27]:


clf.fit(X1_train, y1_train)


# In[28]:


clf.predict(X1_test)


# In[29]:


clf.score(X1_train, y1_train)


# In[30]:


clf.score(X1_test, y1_test)


# Based on the decision tree classification finding, the training and test score were close to each other, so the model could have a high probability of having the correct prediction. Furthermore, we dodn't have to worry about overfitting since the training score was lower than the test score.

# In[31]:


(y1_train == clf.predict(X1_train)).value_counts()


# In[32]:


fig = plt.figure()
_ = plot_tree(clf,
                   feature_names = clf.feature_names_in_,
                   class_names=clf.classes_,
                   filled=True)


# In[33]:


clf.feature_names_in_


# In[34]:


pd.Series(clf.feature_importances_)


# In[35]:


pd.Series(clf.feature_importances_, index=clf.feature_names_in_).sort_values()


# The factors that will most significantly impact the decision tree in the findings of features importance are the higher value numbers. For example, the most relevant factors to predict whether a person has a heart disease or not are Oldpeak and MaxHR.

# ### Random Forest Classifier

# In[36]:


X2 = df[num_cols]
y2 = df["HD"]


# In[37]:


X2_train, X2_test, y2_train, y2_test = train_test_split(X2,y2,  train_size = 0.8, random_state = 1)


# In[38]:


rfc = RandomForestClassifier(n_estimators = 1000, max_leaf_nodes = 10, random_state=1)


# In[39]:


rfc.fit(X2_train, y2_train)


# In[40]:


rfc.score(X2_train, y2_train)


# In[41]:


rfc.score(X2_test, y2_test)


# The scores for random forest are about the same as the decision tree. In the random forest classification, the test and training set scores are relatively close together, and there are no signs of overfitting. The score is also somewhat high, showing that this information can accurately predict heart disease. 

# ### K- Nearest Neighbors Classifier

# In[42]:


X3 = df[num_cols]
y3 = df["HD"]


# In[43]:


X3_train, X3_test, y3_train, y3_test = train_test_split(X3,y3,  train_size = 0.8, random_state = 1)


# In[44]:


clf2 = KNeighborsClassifier(n_neighbors=10)


# In[45]:


clf2.fit(X3_train, y3_train)


# In[46]:


clf2.predict(X3_test)


# In[47]:


clf2.score(X3_test, y3_test)


# In[48]:


clf2.score(X3_train, y3_train)


# The scores for this classifier have a more significant difference than the other scores for the training and test set and are about 6% apart from each other, so it could be a sign of overfitting since the training set score is higher than the test set score. But the scores are also much lower than the others, so this classifier is probably not the best one to use. 

# In[49]:


df["Prediction"] = clf2.predict(X3)
df


# In[50]:


c1 = alt.Chart(df).mark_circle().encode(
    x="Age",
    y="MaxHR",
    color="HD:N"
)
c2 = alt.Chart(df).mark_circle().encode(
    x="Age",
    y="MaxHR",
    color="Prediction:N"
)


# In[51]:


c3 = alt.Chart(df).mark_circle().encode(
    x="Age",
    y="Oldpeak",
    color="HD:N"
)
c4 = alt.Chart(df).mark_circle().encode(
    x="Age",
    y="Oldpeak",
    color="Prediction:N"
)


# In[52]:


c1 | c2


# In[53]:


c3 | c4


# By looking at the graphs, the heart disease predictor had more points for false for heart disease than the actual data did. This is the same for both sets of graphs looking at the highest factors in correlation to heart disease.

# ## Summary
# 
# My project aimed to predict the possibility of a person having a heart condition based on some common risk factors. I used several different classifiers to show the probability of predicting a heart disease with the given data. The classifiers with better scores for the training and test set were the decision tree and random forest classifier, with both having about the same score. Even though the models had a pretty high accuracy score of around 80%, the correlations were not high enough to determine if a person has heart disease based on the factors alone. The highest correlation between heart disease and one of the data set risk factors, Oldpeak, was about 40% which needs to be higher to determine if it correlated to someone having a heart disease. Even though the accuracy was relatively high, there were some false negatives which would have a bad outcome since people will think they don't have heart disease when they do.

# ## References
# 
# Your code above should include references.  Here is some additional space for references.

# * What is the source of your dataset(s)?
# 
# https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction

# * List any other references that you found helpful.
# 
# https://www.geeksforgeeks.org/how-to-create-a-seaborn-correlation-heatmap-in-python/
# https://www.kaggle.com/code/durgancegaur/a-guide-to-any-classification-problem
# https://www.datacamp.com/tutorial/k-nearest-neighbor-classification-scikit-learn
# https://christopherdavisuci.github.io/UCI-Math-10-W22/Week6/Week6-Wednesday.html
# 

# ## Submission
# 
# Using the Share button at the top right, **enable Comment privileges** for anyone with a link to the project. Then submit that link on Canvas.

# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=73a12137-6205-47f9-bda5-430423e844bc' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>
