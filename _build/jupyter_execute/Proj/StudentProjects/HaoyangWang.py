#!/usr/bin/env python
# coding: utf-8

# # The analysis of relations between the total covid-affected population and other datas
# 
# Author: Haoyang Wang
# 
# Course Project, UC Irvine, Math 10, F22

# ## Introduction
# 
# During the pandemic, millions of people get affected by covid-19. This project is for finding the relationship between how many people were affected in total (Total Cases) and other datas (Population, pcr-Test etc.). The project used pandas, altair, seaborn, and machine learning tools in finding the relation.

# ## Explore the datas by using Pandas

# In[1]:


import pandas as pd
import numpy as np
import altair as alt
import seaborn as sns


# First, let's see what the dataset generally looks like.

# In[2]:


df = pd.read_csv("Covid Live.csv")
df.head()


# I noticed that there are something wrong with the columns‘ names for the dataframe. Therefore, for calling the columns easily, I rename the dataframe.

# In[3]:


# Call columns to make sure store their true name in the dictionary later
df.columns


# In[4]:


# Use the dictionary help me with renaming
col_name = {'Country,\nOther':"Country", 'Total\nCases':"Total Cases", 'Total\nDeaths':"Total Deaths", 'New\nDeaths':"New Deaths",
       'Total\nRecovered':"Total Recovered", 'Active\nCases':"Active Cases", 'Serious,\nCritical':"Serious Criticals",
       'Tot Cases/\n1M pop':"Total Cases/m", 'Deaths/\n1M pop':"Deaths/m", 'Total\nTests':"Total Test",
       'Tests/\n1M pop':"Tests/m"}


# In[5]:


df.rename(col_name, axis=1, inplace=True)
df.head()


# Drop missing values and the whole "New Deaths" column because it almost have no data.

# In[6]:


del df["New Deaths"] #"del" reference: https://www.educative.io/answers/how-to-delete-a-column-in-pandas
df = df.dropna(axis=0).copy()
df.head()


# I noticed that the data in the columns are strings with comma. I need to convert them into numeric type for manipulating easier.

# In[7]:


# first, find the columns who are string
from pandas.api.types import is_string_dtype
str_col = [i for i in df.columns if is_string_dtype(df[i])]

# the first string column is "Country", drop it
str_col = str_col[1:]


# In[8]:


# Delete the commas, then convert the string columns into numberic
for i in str_col:
    df[i] = df[i].str.replace(",","")
    df[i] = pd.to_numeric(df[i])


# ## Visualization by Altair and Seaborn

# First of all, I'd like to see whether the population and the density (total cases/million people) of affected patients have a strong relation. In other words, is the population affect the possibility of being affected.

# In[9]:


alt.Chart(df).mark_circle().encode(
    x = alt.X("Population",sort="ascending"),
    y = alt.Y("Total Cases/m",sort="ascending"),
    tooltip = ("Country","Total Cases")
).interactive()


# I use selection for showing the details.

# In[10]:


sel = alt.selection_interval()

c1 = alt.Chart(df).mark_circle().encode(
    x =alt.X("Population", scale=alt.Scale(domain=[0,400000000])),
    y ="Total Cases/m",
    tooltip = ("Country","Total Cases"),
    color = alt.condition(sel, alt.value("Black"), "Population")
).add_selection(sel)

c2 = alt.Chart(df).mark_bar().encode(
    x ="Country",
    y ="Total Cases/m", 
    color ="Country"
).transform_filter(sel)

alt.vconcat(c1,c2) # Can choose a rectangle by mouse to see the datas specifically


# Without considering the outliars (China and India), we could say that it is hard to observe the strong relation between the density of affected patients and the population. Thus, I'd like check whether Test helps in preventing people from affected.

# In[11]:


sns.scatterplot(
    data = df,
    x = "Tests/m",
    y = "Total Cases/m"
)


# Surprisingly, a negative relation (which impies tests do prevent people from affected) is not observed here between Tests/m and Total Cases/m.

# ## DecisionTreeRegressor

# Though the relations between other datas and "Total Tests" are hard to be observed, I'm going to use machine learning to try to predict the "Total Tests" by using other datas as input.

# For avoiding repeat affection on the prediction result, I will delete some of the inputs which are repeated, for example, "Total Test", and "Tests/m".

# In[12]:


cols = [i for i in str_col if i[-2:] != "/m"]
cols = cols[1:] #the first one is "Total Cases" (predict result), drop it


# In[13]:


from sklearn.model_selection import train_test_split
X = df[cols]
y = df["Total Cases/m"]
X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=.8,random_state=59547172)


# In[14]:


from sklearn.tree import DecisionTreeRegressor
reg = DecisionTreeRegressor()
reg.fit(X_train, y_train)


# In[15]:


reg.score(X_test, y_test) #the predicted result not seems good, might be overfitting


# In[16]:


reg.score(X_train, y_train) # 1.0 implies there is overfitting


# the feature importances seem not really make sense to me

# In[17]:


pd.Series(reg.feature_importances_, index=reg.feature_names_in_)


# To make sure I use a proper max_leaf_nodes number, I will make a U-shape test error curve first.

# In[18]:


from sklearn.metrics import mean_absolute_error


# In[19]:


train_dic = {}
test_dic = {}

for i in range(2,100):
    reg = DecisionTreeRegressor(criterion = "absolute_error", 
    max_leaf_nodes = i)
    reg.fit(X_train, y_train)
    train_dic[i] = mean_absolute_error(y_train, reg.predict(X_train))
    test_dic[i] = mean_absolute_error(y_test, reg.predict(X_test))


# In[20]:


train_loss = pd.Series(train_dic)
test_loss = pd.Series(test_dic)
train_loss.name = "train"
test_loss.name = "test"
df_loss = pd.concat((train_loss, test_loss), axis=1)
df_loss.reset_index(inplace=True)
df_loss.rename({"index": "max_leaf_nodes"}, axis=1, inplace=True)
df_melted = df_loss.melt(id_vars="max_leaf_nodes", var_name="error_type", value_name="loss")


# The curve is not typically U-shaped, and the best max_leaf_nodes here I can choose where is not overfitting is 17. (Actually, I also check 37, but the score of train data is .98 versus .70 test score, which implies overfitting, so I just discard 37.)

# In[21]:


alt.Chart(df_melted).mark_line().encode(
    x = "max_leaf_nodes",
    y = "loss",
    color = "error_type",
    tooltip = "max_leaf_nodes"
)


# Do the DicisionTreeRegressor again.

# In[22]:


reg2 = DecisionTreeRegressor(max_leaf_nodes=17)


# In[23]:


reg2.fit(X_train, y_train)


# In[ ]:


reg2.score(X_test, y_test) # the score still not perform well


# ## RandomForestRegressor

# I will use randomforest to make the predict more accurate.

# In[26]:


from sklearn.ensemble import RandomForestRegressor


# In[27]:


forest_reg = RandomForestRegressor(n_estimators=100, max_leaf_nodes=17)


# In[28]:


forest_reg.fit(X_train, y_train)


# In[29]:


forest_reg.score(X_test, y_test)


# In[30]:


df["pred"] = forest_reg.predict(df[cols])


# In[31]:


c1 = alt.Chart(df).mark_circle(color="black").encode(
    x = alt.X("Population", scale=alt.Scale(domain=[0,400000000])),
    y = "Total Cases",
    tooltip = "Country"
)

c2  = alt.Chart(df).mark_circle(color="red").encode(
    x = alt.X("Population", scale=alt.Scale(domain=[0,400000000])),
    y = "pred"
)

c1+c2


# The chart is not performing good, I think the outliars make a lot effects on the predicted result.

# ## KNeighborsRegressir

# I will use KNeighborsRegressor to check the predict result again.

# In[32]:


from sklearn.neighbors import KNeighborsRegressor


# In[33]:


# Again, start with finding the best k
def get_scores(k):
    K_reg=KNeighborsRegressor(n_neighbors=k)
    K_reg.fit(X_train, y_train)
    train_error=mean_absolute_error(K_reg.predict(X_train), y_train)
    test_error=mean_absolute_error(K_reg.predict(X_test), y_test)
    return (train_error, test_error)


# In[34]:


df_k = pd.DataFrame(columns=("train_error", "test_error"))


# In[35]:


df_k["train_error"] = [get_scores(k)[0] for k in range(1,100)]
df_k["test_error"] = [get_scores(k)[1] for k in range(1,100)]
df_k["k"] = df_k.index


# By the chart, we know higher the k, bigger the error. The best K I can get from it is 5

# In[36]:


sns.lineplot(data=df_k, markers=True)


# In[37]:


K_reg = KNeighborsRegressor(n_neighbors=5)
K_reg.fit(X_train, y_train)
df["K_predict"] = K_reg.predict(df[cols])


# Still, by using KNeighbors, the predict results do not vary from randomforest a lot. Therefore, I conclude that there is no much relations between the Total Cases and other datas.

# In[38]:


c3 = alt.Chart(df).mark_circle(color="black").encode(
    x = alt.X("Population", scale=alt.Scale(domain=[0,400000000])),
    y = "Total Cases",
    tooltip = "Country"
)

c4  = alt.Chart(df).mark_circle(color="red").encode(
    x = alt.X("Population", scale=alt.Scale(domain=[0,400000000])),
    y = "K_predict"
)

c3+c4


# ## Summary
# 
# I use pandas in cleaning and analyzing datas, altair/seaborn for visualization, and Decision Tree/Random Forest/KNeighbors in machine learning. Though I used different techniques, the predicted results do not perform well. Therefore, I would say it is hardly to oberserve a relationship between the Total Cases with other datas based on this dataframe.

# ## References
# 
# del: "How to delete a column in pandas" by Neko Yan, https://www.educative.io/answers/how-to-delete-a-column-in-pandas
# 
# Seaborn Visualization: Seaborn.pydata
# https://seaborn.pydata.org/generated/seaborn.scatterplot.html
# https://seaborn.pydata.org/generated/seaborn.lineplot.html

# * What is the source of your dataset(s)?
# Kaggle

# * List any other references that you found helpful.
# KNeighborsRegressor: Chris's previous lecture
# https://christopherdavisuci.github.io/UCI-Math-10-W22/Week6/Week6-Wednesday.html

# ## Submission
# 
# Using the Share button at the top right, **enable Comment privileges** for anyone with a link to the project. Then submit that link on Canvas.

# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=ed82ad73-cd7d-47c0-8299-ffa3259fcf6a' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>
