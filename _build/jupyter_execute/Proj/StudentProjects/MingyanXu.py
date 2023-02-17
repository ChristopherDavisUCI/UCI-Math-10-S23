#!/usr/bin/env python
# coding: utf-8

# # The relationship and prediction between Pokemon's stats and generation
# 
# Author: Mingyan Xu
# 
# Course Project, UC Irvine, Math 10, F22

# ## Introduction
# 
# Introduce your project here.  Maybe 3 sentences.
# 
# From other's project in the past quarter, I found that there were some analysis about Pokemon, which was one of my favorite games, so I chose to explore more about this dataset. The main topic I choose is to find the relationship between the generations and the each stats for pokemons, and study whether the strength of Pokemon has become stronger through generations. In addition, I will try to predict the trend on designing new generation pokemon, such as their type and stats.

# ## Import Section

# In[1]:


import pandas as pd
import altair as alt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# ## Feature Engineering

# This dataset contains the general information of 800 different pokemons, including their names, types, HP, attack, defense and other stats.

# In[2]:


df = pd.read_csv("Pokemon.csv")
df


# I think I'm going to use the pokemons that have two types to avoid the error porvided by types. 
# Use dropna() method to drop the nan value in the whole dataset.

# In[3]:


df = df.dropna()


# I plan to use non-legendary pokemon for this project because the legendary pokemons are designed to have higher stats compared to the regular pokemons.

# In[4]:


df = df[df["Legendary"]==False]
df


# I don't need the "Legendary" column anymore, so I just drop it.

# In[5]:


df = df.drop("Legendary",axis=1)


# The original column names are not working when doing the altair chart, so I change  the name for speical attack and defense.

# In[6]:


df = df.rename(columns={"Sp. Atk":"SpecialA","Sp. Def": "SpecialD"})


# In[7]:


df.shape


# ## Analysis of the dataset

# To focus on the generation, first check the number of pokemons in each generations.

# In[8]:


df["Generation"].value_counts()


# Use groupby method to see more detailed information.

# In[9]:


df.groupby("Generation").apply(display)


# Divide the original dataset into 6 based on generations.

# In[10]:


grouped = df.groupby("Generation")
list=[]

for gen, group in grouped:
    list.append(group)


# ## Data Visualization

# First visualizing the number of pokemons in each generation and find that the first generation has the most and the sixth generation has the least.

# In[11]:


alt.Chart(df).mark_bar(size=15).encode(
    x=alt.X("Generation"),
    y="count()",
    color="Generation:N"
)


# In[51]:


plt.figure(figsize=(6,4))
corr=df.iloc[:,5:11].corr().round(3)
sns.heatmap(corr,annot=True)
sns.set(font_scale=1.0)
sns.set_style("whitegrid")


# In[13]:


alt.Chart(df).mark_circle().encode(
    x=alt.X("SpecialA",scale=alt.Scale(domain=[0, 250])),
    y=alt.Y("Attack",scale=alt.Scale(domain=[0, 240])),
    color="Generation:N",
    tooltip=["Name","Attack","SpecialA"]
).properties(
    title="Attack v.s. Special Attack based on generation"
)


# In[14]:


alt.Chart(df).mark_circle().encode(
    x=alt.X("SpecialD",scale=alt.Scale(domain=[0, 250])),
    y=alt.Y("Defense",scale=alt.Scale(domain=[0, 240])),
    color="Generation:N",
    tooltip=["Name","Defense","SpecialD"]
).properties(
    title="Defense v.s. Special Defense based on generation"
)


# From the chart above, we can find that the general shape and trend for each generation are similar. 
# I find that the distribution is not similar to what I believe before. I think a pokemon should have whether a high attack or high special attack, but the graph shows that most of the pokemon have the same attack and special attack. Also the same observation for the defense and special defense.

# ## K-Mean clusters
# 
# Use the stats value to predict clusters. I choose 6 because there are total 6 generations in the dataset.

# In[15]:


kmeans = KMeans(n_clusters=6)


# Use all the stats value in the dataset to predict the cluster.

# In[16]:


first_col = "Total"
last_col = "Speed"


# In[17]:


kmeans.fit(df[[first_col,last_col]])


# In[18]:


arr = kmeans.predict(df[[first_col,last_col]])


# In[19]:


df["cluster"]= arr


# In[20]:


alt.Chart(df).mark_circle().encode(
    x="Attack",
    y="SpecialA",
    color="cluster:N",
    tooltip=["#","Name","Attack","SpecialA"]
).facet(
    row="Generation"
)


# The charts show that the predicting cluster is roughly and evenly distributed in each generation, so we can get a conclusion that Pokemon's stats strength are not clearly related to generation.
# Therefore, we can say that The number of Pokemons of similar strength is about the same in each generation based on the result from K-means cluster.

# ## Machine Learning - Linear Regression
# 
# For the machine learning part, I first choose to use the linear regression model to predict the generation based on the known data. 

# In[21]:


features = ["Total","HP","Attack","SpecialA","Defense","SpecialD","Speed"]


# In[22]:


X_train, X_test, y_train, y_test = train_test_split(df[features],df["Generation"],train_size=0.7)


# In[23]:


lin = LinearRegression()


# In[24]:


lin.fit(X_train,y_train)


# In[25]:


linear_train_accuracy = lin.score(X_train,y_train)


# In[26]:


linear_test_accuracy = lin.score(X_test,y_test)


# In[27]:


print(f"When using the linear regression model,\nthe accuracy for the training set is {linear_train_accuracy},\nand the accuracy for the test set is {linear_test_accuracy}.")


# By using the score method, I calculate the accuracy for both train and test data, but the accuracies are very low compared with the dataset we used in class, which means that my dataset doesn't perform well under the linear regression model.

# ## Machine Learning - Logistic Regression
# 
# Then, I try to use the logistic regression model to check if the generation can be predicted using pokemon's stats. 
# For this model, I also include the calculation of the mean squared error to see the accuracy.

# In[28]:


lgr = LogisticRegression(max_iter=500)


# In[29]:


lgr.fit(X_train,y_train)


# In[30]:


log_test_accuracy = lgr.score(X_test, y_test)


# In[31]:


log_train_accuracy = lgr.score(X_train, y_train)


# In[32]:


log_train_error = mean_squared_error(y_train,lgr.predict(X_train))
log_test_error = mean_squared_error(y_test,lgr.predict(X_test))


# In[33]:


print(f"The mean squared  error for the training set is {log_train_error},\n  and the mean squared error for the test set is {log_test_error}.\nThe accuracy for the training set is {log_train_accuracy},\n  and the accuracy for the test set is {log_test_accuracy}.")


# I get the similar result as the linear regression model, even a little bit less accurate because my accuracy is lower and error is relatively high. 

# It is clear that the generation can't be predicted by the stats from three above regression model, so I changed my goal to using attack to predict the special attack for each pokemon for the following regression.
# 

# ## Machine Learning - Decision Tree Regressor
# Since my expectation is to predict the generation, using tree regressor instead of classifier is better.

# In[34]:


alt.Chart(df).mark_line().encode(
    x="Attack",
    y="SpecialA"
)


# There are too many peak in the dataset, maybe it's hard to predict.
# I choose the generation 2, which consists of the most pokemons, and try to predict the stat of speical attacks.

# In[35]:


df1 = list[1]


# In[36]:


reg = DecisionTreeRegressor(max_leaf_nodes=10,max_depth=10)


# In[37]:


reg.fit(df1[["Attack"]],df1["SpecialA"])


# In[38]:


df1["PredictA"] = reg.predict(df1[["Attack"]])


# In[39]:


d1 = alt.Chart(df1).mark_line().encode(
    x="Attack",
    y="SpecialA"
)

d2 = alt.Chart(df1).mark_line(color="red").encode(
    x="Attack",
    y="PredictA"
)
d1+d2


# The graph shows that the prection is kind of accurate, but there is a big peak in the middle of the data which is hard to predect.

# ## Machine Learning - Random Forest Regressor
# 
# Finally, try to use random forest regressor to predict pokemons in generation 6, which has the least pokemons.

# In[40]:


df2 = list[5]


# In[41]:


features2 = ["Total","HP","Attack","Defense","SpecialD","Speed"]


# In[42]:


X_train1, X_test1, y_train1, y_test1 = train_test_split(df2[features2],df2["SpecialA"],train_size=0.6,random_state=32203564)


# In[43]:


rfe = RandomForestRegressor(n_estimators=100, max_leaf_nodes=15)


# In[44]:


rfe.fit(X_train1,y_train1)


# In[45]:


rfe.score(X_train1,y_train1)


# In[46]:


rfe.score(X_test1,y_test1)


# ### Graphing for the Random Forest Regressor

# In[47]:


rfe.fit(df2[["Attack"]],df2["SpecialA"])


# In[48]:


df2["PredictA"] = rfe.predict(df2[["Attack"]])


# In[49]:


d1 = alt.Chart(df2).mark_line().encode(
    x="Attack",
    y="SpecialA"
)

d2 = alt.Chart(df2).mark_line(color="red").encode(
    x="Attack",
    y="PredictA"
)
d1+d2


# ## Summary
# 
# Either summarize what you did, or summarize the results.  Maybe 3 sentences.

# I try to use different methods, including K-Mean clusters,logistics regression and linear regression, to analysis the relationship between generations and the stats of pokemons, but none of them performed very well to fit the data. Also, it is hard to do the prediction for future generation pokemons based on the stats becasue the relationship is not strong enough. 
# After failing to predict the generation, I try to predict the special attack values based on attack values using dicision tree regressor and random forest regressor, and find that the special attack values can be predected by the attack values with small error and high accuracy.

# ## References
# 
# Your code above should include references.  Here is some additional space for references.

# * What is the source of your dataset(s)?
# My dataset is from Kaggle. [Pokemon](https://www.kaggle.com/datasets/abcsds/pokemon)

# * List any other references that you found helpful.
# [Tutorial for Pokemon](https://www.kaggle.com/code/christinobarbosa/machinelearningmodel-pokemon/notebook)
# [Pinting and forming new dataframe based on a groupby object](https://stackoverflow.com/questions/22691010/how-to-print-a-groupby-object)
# [Difference between Dicision Tree Regressor and Classifier](https://becominghuman.ai/machine-learning-series-day-6-decision-tree-regressor-82a2e2f873a
# )

# ## Submission
# 
# Using the Share button at the top right, **enable Comment privileges** for anyone with a link to the project. Then submit that link on Canvas.

# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=259c7c54-5532-4802-8a81-ac7f2e4dd773' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>
