#!/usr/bin/env python
# coding: utf-8

# # Netflix Stock Price
# 
# Author: Huangxiao Zhang
# 
# Course Project, UC Irvine, Math 10, F22

# ## Introduction
#  In this predicting project, I am going to make a prediction of Netflix Stock price since I am a big fan of this comany. The data is the price for netflix stock from 2002 to 2021. In the project, I am going to use linear regression and decision tree's Predicted values compare with truth value.
# 

# ## Importing data

# In[1]:


import numpy as np
import pandas as pd
import altair as alt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import seaborn as sns


# In[2]:


df = pd.read_csv('netflix.csv')


# ## Sorting Data

# ### Original Dataset

# In[3]:


df


# Rename Adj Close to Adjusted Closing Price.([Definition](https://help.yahoo.com/kb/SLN28256.html?guccounter=1) of adjusted closing price)

# In[4]:


df = df.rename(columns={'Adj Close' : 'Adjusted Closing Price'}) 


# Change type from object to datetime64[ns]

# In[5]:


df["Date"] = pd.to_datetime(df["Date"])


# Clean null value

# In[6]:


df.dropna(inplace=True)


# ### Clean Dataset

# In[7]:


df


# ## Seaborn and Altair Chart

# This is the line chart which shows the Adjusted Closing Price increase with time by [sns.lineplot](https://seaborn.pydata.org/examples/errorband_lineplots.html).

# In[18]:


sns.set_theme(style="whitegrid")
sns.lineplot(x="Date", y="Adjusted Closing Price", data=df).set(title='Date and Adjusted Closing Price')


# This is a bar chart which indicates the changes of volume is cyclical

# In[9]:


chart = alt.Chart(df).mark_bar().encode(
    x='Date',
    y='Open',
).properties(
    title='Date and Open Price'
)
chart


# ## Build Training and Test Set
# 

# Split the dataset into 2 parts: X includes Highest Price, Lowest Price, Openning Price, and Volume; y includes Adjusted Closing Price.

# In[10]:


X = df.loc[:, ["High", "Low", "Open", "Volume"]]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)


# ## Linear Regression

# Train the data

# In[11]:


reg = LinearRegression()
reg.fit(X_train, y_train)


# predicting value for Linear Regression

# In[12]:


linear_pred = reg.predict(X_test)


# find the mean squared error for Linear Regression

# In[13]:


linear_mse = mean_squared_error(y_test, linear_pred)
linear_mse


# ## Decision Tree Regressor

# Train the data

# In[14]:


dt = DecisionTreeRegressor()
dt.fit(X_train, y_train)


# predicting value for DecisionTree

# In[15]:


dt_pred = dt.predict(X_test)


# find the mean squared error for DecisionTree

# In[16]:


dt_mse = mean_squared_error(y_test, dt_pred)
dt_mse


# ## Result

# Comparing Truth value and Predicted value within one chart. [sns.scatterplot](https://seaborn.pydata.org/examples/different_scatter_variables.html)

# In[17]:


sns.set_theme()
results = pd.DataFrame({
    "Type": ["Linear"]*y_test.shape[0] + ["DT"]*y_test.shape[0], 
    "Truth": y_test.tolist() * 2,
    "Pred": linear_pred.tolist() + dt_pred.tolist()
})

sns.scatterplot(x="Truth", y="Pred", hue="Type", data=results, alpha=0.5, s=9).set(title='Truth vs. Pred for Linear and DT')


# ## Summary
# At the beginning, I plot two images to show the changes of Adjusted Closing Price and volume. The first image is a line chart which shows the Adjusted Closing Price increase with time while the second image is a bar chart which indicates the changes of volume is cyclical. Next I split the dataset into 2 parts, one with 80% random samples as train set and the remaine 20% random samples as test set. I fit a linear regression model and a decision model based on train set and evaluate the performances of them by these sets. The results shows that the mean squared error of the linear model is 2.0652666805811815 while the mean squared error of the decision tree is 8.75387563145938. Finally I plot a scatterplot which the x axis stands for the value of Adjusted Closing Price in the test set and the y axis represents the value of predictions of the two models. The scatterplot shows that both models has a relative wonderful performances.
# 

# ## References
# 
# Your code above should include references.  Here is some additional space for references.
# [sns.lineplot](https://seaborn.pydata.org/examples/errorband_lineplots.html)
# [sns.scatterplot](https://seaborn.pydata.org/examples/different_scatter_variables.html)
# 

# * What is the source of your dataset(s)?
# [Kaggle](https://www.kaggle.com/datasets/akpmpr/updated-netflix-stock-price-all-time)

# * List any other references that you found helpful.
# [seaborn.set_theme](https://seaborn.pydata.org/generated/seaborn.set_theme.html#seaborn.set_theme)

# ## Submission
# 
# Using the Share button at the top right, **enable Comment privileges** for anyone with a link to the project. Then submit that link on Canvas.

# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=42fe845b-b200-408b-92a9-f532cc933eba' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>
