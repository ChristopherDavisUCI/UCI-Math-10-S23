#!/usr/bin/env python
# coding: utf-8

# # Replace this with your project title
# 
# Author:Derek Jiang
# 
# Course Project, UC Irvine, Math 10, F22

# ## Introduction
# 
# In this project, I will be using different stock prices and volume inside the dataset to predict different gas companies in the US, specifically, from 2015 to 2022. I will also perform some machine learning algorithms to find out which information play the most important role in determining the gas company names. Additionally, I will explore the relationship between these prices during the Covid Pandemic using different graphs and find out some interesting features.

# ## Preparation
# In this section, I will perform basic data cleaning and sorting so I can obtain the data from 2015 to 2022
# `

# In[1]:


import pandas as pd
import numpy as np
import altair as alt
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from pandas.api.types import is_numeric_dtype

alt.data_transformers.enable('default', max_rows=20000)


# In[2]:


df = pd.read_csv("oil and gas stock prices.csv")
df.isna().any(axis=0)


# We performed the above analysis to ensure there is any missing value that needs to be dropped in the dataset. We will also rename column "Symbol" to "Company" for better understanding.

# In[3]:


df.rename(columns={"Symbol":"Company"},inplace=True)


# In[4]:


df["Year"] = pd.to_datetime(df["Date"]).dt.year
df


# In[5]:


df_sub = df[df["Year"]>=2015].copy()
df_sub


# In[6]:


df1= df_sub[df_sub["Year"]<2020].copy()
df2= df_sub[df_sub["Year"]>=2020].copy()


# df1 and df2 are created for future use and reference.

# ## Use DecisionTreeClassifier to Predict

# Here is the process of using DecisionTreeClassifier algorithm to predict the company names.

# In[7]:


col = list(c for c in df.columns if is_numeric_dtype(df[c]) and c != "Year")
col


# In[8]:


X = df_sub[col]
y = df_sub["Company"]


# In[9]:


X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.9,random_state=0)


# In[10]:


train_dict={}
test_dict={}

for n in range (10,300,10):
    clf = DecisionTreeClassifier(max_leaf_nodes=n, max_depth=10)
    clf.fit(X_train,y_train)
    c1 = clf.score(X_train,y_train)
    train_error = log_loss(y_train,clf.predict_proba(X_train))
    train_dict[n]=train_error
    c2 = clf.score(X_test,y_test)
    test_error = log_loss(y_test,clf.predict_proba(X_test))
    test_dict[n]=test_error
    print(c1,c2)


# After getting the Classifier score, it is actually hard to tell if there is overfitting since testing score lingers between 0.59 and 0.6, and there is no decreasing trend. Therefore, we will use log_loss to decide.

# In[11]:


train_dict


# In[12]:


test_dict


# From the log_loss result, we can see that after n=70, the testing error starts increasing drastically, so overfitting occurs after n=70.

# In[13]:


clf=DecisionTreeClassifier(max_leaf_nodes=70, max_depth=10)


# In[14]:


clf.fit(X_train,y_train)


# In[15]:


clf.score(X_train,y_train)


# In[16]:


clf.score(X_test,y_test)


# In[17]:


arr = clf.predict(X_train)


# In[18]:


fig = plt.figure()
_= plot_tree(clf,
                  feature_names=clf.feature_names_in_,
                  class_names=clf.classes_,
                  filled = True)


# ## Use K-Nearest Neighbors Classifier to Predict
# 

# After using DecisionTreeClassifier, I want to see if there is a better Classifier algorithm that can be applied to this dataset, and I decide to try KNeighbor method with the guidance of additional classnotes

# In[19]:


kcl = KNeighborsClassifier(n_neighbors=10)


# In[20]:


kcl.fit(X_train,y_train)


# In[21]:


kcl.score(X_train,y_train)


# In[22]:


def kscore(k):
    kcl = KNeighborsClassifier(n_neighbors=k)
    kcl.fit(X_train,y_train)
    a = kcl.score(X_train,y_train)
    b = kcl.score(X_test,y_test)
    return (a,b)


# In[23]:


[kscore(10),kscore(20),kscore(30),kscore(40)]


# As we see from the result, the test score is considerably smaller than that of DecisionTreeClassifier, and so the prediction is not as accurate as DecisionTreeClassifier  

# ## Using DecisionTreeClassifier on df1

# Now I will use DecisionTreeClassifier on datasets before and after the pandemic occured, and I want to find out two most important features in predicting the companies.

# In[24]:


df1


# In[25]:


X1 = df1[col]
y1 = df1["Company"]


# In[26]:


X_train, X_test, y_train, y_test = train_test_split(X1,y1,train_size=0.9,random_state=1)
clf1 = DecisionTreeClassifier(max_depth=9)
clf1.fit(X_train,y_train)


# In[27]:


clf1.score(X_train,y_train)


# In[28]:


clf1.score(X_test,y_test)


# In[29]:


pd.Series(clf1.feature_importances_,index= clf1.feature_names_in_)


# ## Using DecisionTreeClassifier on df2

# In[30]:


df2


# In[31]:


X2 = df2[col]
y2 = df2["Company"]


# In[32]:


X_train, X_test, y_train, y_test = train_test_split(X2,y2,train_size=0.9,random_state=1)
clf2 = DecisionTreeClassifier(max_depth=5)
clf2.fit(X_train,y_train)


# In[33]:


clf2.score(X_train,y_train)


# In[34]:


clf2.score(X_test,y_test)


# In[35]:


pd.Series(clf2.feature_importances_,index= clf2.feature_names_in_)


# ## High Price and Volume Relationship
# Now we have seen that High Price and Volume play the two most important roles in predicting the companies. We want to explore the relationship between these two. 

# In[36]:


c1 = alt.Chart(df1).mark_circle().encode(
    x = "High",
    y = "Volume",
    color = "Company",
    tooltip =["Date","High","Volume"],
).properties(
    title = "Before Covid"
)

c2 = alt.Chart(df2).mark_circle().encode(
    x = "High",
    y = "Volume",
    color = "Company",
    tooltip =["Date","High","Volume"],
).properties(
    title = "During Covid"
)


# In[37]:


alt.concat(c1,c2)


# From the graphs, it is clear that before covid happened, Volume values stay relatively low no matter how High price changes. After covid, Volume drastically rised up, but we cannot exactly explain the relationship yet.

# In[38]:


sel = alt.selection_single(fields=["Company"],bind="legend")
c3 = alt.Chart(df2).mark_circle().encode(
    x = "High",
    y = "Volume",
    color = alt.condition(sel,"Company",alt.value("lightgrey")),
    opacity =alt.condition(sel,alt.value(1),alt.value(0.1)),
    tooltip =["Date","High","Volume"],
).properties(
    title = "During Covid"
).add_selection(sel)
c3


# By using selection on Company, every company's data point can be seen in the graph, but it is still not the best visual effect. 

# In[39]:


c4 = alt.Chart(df2).mark_circle().encode(
    x = "High",
    y = "Volume",
    color = "Year",
    tooltip =["Date","High","Volume"],
).properties(
    height = 200,
    width = 200,
).facet(
    column = "Company"
)
c4


# Using Facet gives a very clear view of each company's volume and high price relationship from 2020-2022. The majority of the companeis stock volume were high at low high price at 2020, and as time went on and high price went up, volume stayed roughly the same or gradually decreased.

# Now, we want to use the predicted Companies to plot and see the relationship between High Price and Volume in df2

# In[40]:


df3 = pd.DataFrame()
df3["Pred"]=clf2.predict(X_test)


# In[41]:


h1=[]
h2=[]
h3=[]
h4=[]
h5=[]
h6=[]
h7=[]
for i in X_test.index:
    h1.append(df2.loc[i,"Date"])
    h2.append(df2.loc[i,"Open"])
    h3.append(df2.loc[i,"High"])
    h4.append(df2.loc[i,"Low"])
    h5.append(df2.loc[i,"Close"])
    h6.append(df2.loc[i,"Volume"])
    h7.append(df2.loc[i,"Company"])


# In[42]:


df3["Date"]=h1
df3["Open"]=h2
df3["High"]=h3
df3["Low"]=h4
df3["Close"]=h5
df3["Volume"]=h6
df3["Company"]=h7
df3


# In[43]:


sel = alt.selection_single(fields=["Pred"])
c5 = alt.Chart(df3).mark_line().encode(
    x = "High",
    y = "Volume",
    color = alt.condition(sel,"Pred",alt.value("lightgrey")),
    opacity =alt.condition(sel,alt.value(1),alt.value(0.1)),
    tooltip =["Date","High","Volume"],
).properties(
    title = "During Covid(Predicted)"
).add_selection(sel)

c6 = alt.Chart(df3).mark_line().encode(
    x = "High",
    y = "Volume",
    color = alt.condition(sel,"Company",alt.value("lightgrey")),
    opacity =alt.condition(sel,alt.value(1),alt.value(0.1)),
    tooltip =["Date","High","Volume"],
).properties(
    title = "During Covid(Original)"
).add_selection(sel)

alt.concat(c5,c6)


# Comparing the predicted one with original one, the overall trend still matched well. The accuracy can be calculated as below, which corresponds to the test score. And the highest volume occured on 03/09/2022 on both graphs.

# In[44]:


(df3["Pred"]==df3["Company"]).sum()/df3.shape[0]


# ## Additional Findings
# After seeing the relationship between Volume and High Price, I want to use another way to show the date on which the highest volume occured and explore which High Price occured the most frequently in df2. To do this, I will include some codes reference learned from Kaggle.

# In[45]:


import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


# In[46]:


fig, ax = plt.subplots(figsize = (15, 6))
ax.bar(df2["Date"], df2["Volume"])
ax.xaxis.set_major_locator(plt.MaxNLocator(15))
ax.set_xlabel("Date", fontsize = 10)
ax.set_ylabel("Volume", fontsize = 10)
plt.title('Volume Trend', fontsize = 20)
plt.grid()
plt.show()


# The above graph is a volume trend of original df2 data, and we can see that one top volume occured slightly before 03/16/2020, which still corresponds to the above result found in the graph 

# In[47]:


plt.figure(figsize = (12, 6))
sns.distplot(df2["High"], color= "#FFD500")
plt.title("Distribution of open prices of US Oil and Gas stocks", fontweight = "bold", fontsize = 20)
plt.xlabel("High Price", fontsize = 10)

print("Maximum High price of stock ever obtained:", df2["High"].max())
print("Minimum High price of stock ever obtained:", df2["High"].min())


# In[48]:


plt.figure(figsize = (12, 6))
sns.distplot(df3["High"], color= "#FFD500")
plt.title("Distribution of open prices of US Oil and Gas stocks", fontweight = "bold", fontsize = 20)
plt.xlabel("High Price", fontsize = 10)

print("Maximum High price of stock ever obtained:", df3["High"].max())
print("Minimum High price of stock ever obtained:", df3["High"].min())


# From both distribution graphs, we can see that the High Price that occurs the most stays around 60. The predicted graph does a good job showing this interesting fact.

# ## Summary
# 
# Either summarize what you did, or summarize the results.  Maybe 3 sentences.

# DecisionTreeClassifier, as a more accurate algorithm compared with K-Nearest Neighbors Classifier, helped me predict the companies using different stock prices and volume. Through exploring the relationship between High Price and Volume, I came to see both overall trend and individual trend of each company. Besides, I found out the date on which the highest volume occurs and the most frequently occured High Price.

# ## References
# 
# Your code above should include references.  Here is some additional space for references.

# * What is the source of your dataset(s)?

# This Dataset is from Kaggle: https://www.kaggle.com/datasets/prasertk/oil-and-gas-stock-prices

# * List any other references that you found helpful.

# Reference 1: https://christopherdavisuci.github.io/UCI-Math-10-W22/Week6/Week6-Wednesday.html
# This is used as a guide to predict companies in the dataset
# 
# Reference 2: https://www.kaggle.com/code/mattop/major-us-oil-and-gas-stock-price-eda 
# I borrowed the codes of volume trending and the Distribution of High Price to graph 

# ## Submission
# 
# Using the Share button at the top right, **enable Comment privileges** for anyone with a link to the project. Then submit that link on Canvas.

# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=3b82a602-1ff4-4142-b60b-fd908347f8fd' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>
