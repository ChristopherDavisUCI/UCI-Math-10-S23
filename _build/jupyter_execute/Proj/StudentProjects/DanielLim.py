#!/usr/bin/env python
# coding: utf-8

# # Spotify Popular Songs Correlation
# 
# Author: Daniel Lim, djlim2@uci.edu
# 
# Course Project, UC Irvine, Math 10, F22

# ## Introduction
# 
# This dataset contains a wide range of songs from Spotify. There are columns describing the artist of the track, popularity of song, how long it is, etc. My goal is to see if there is a correlation between the popularity of the song and song genre using factors such as energy, valence, and danceability.  

# ## Cleaning Up Data
# 
# We first need to clean up the data and get rid of columns that won't be of use to us as well as getting rid of certain missing values. 

# In[1]:


import pandas as pd
import altair as alt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import mean_absolute_error


# In[2]:


df = pd.read_csv('dataset.csv')
df


# Since most of the columns that we want to look at are already integers, we don't need to convert them

# In[3]:


df.columns, df.dtypes


# Dropping these columns in the beginning since they have nothing to do with what I want to do

# In[4]:


df.drop(['Unnamed: 0', 'track_id'], axis=1, inplace=True)


# In[5]:


df


# Checking for any missing values in this columns specifically since a track without an artist makes no sense

# In[6]:


df['artists'].isnull().any().sum() 


# In[7]:


df[df['artists'].isnull()]


# In[8]:


df.drop([65900], inplace=True) # dropped row with missing value 


# In[9]:


df


# In[10]:


df.columns


# I'm grouping the 'track_genre' column with the 'popularity' column just to get an idea of what the most popular genres are and computing the mean of the two. 

# In[11]:


pop_score = df.groupby('track_genre')['popularity'].mean().sort_values(ascending=False)
pop_score


# In[12]:


df = df[['artists', 'track_name', 'popularity', 'danceability', 'energy', 'valence', 'track_genre']]
df # using boolean indexing to keep the columns we want


# ## Main Portion of Project

# The reason I wanted to take a sample of the max number of rows is because I wanted to see if the outcome would be the same or similar regardless of how many times I ran the code (I tested it about 5 times and the result seems to be similar)

# In[13]:


df = df.sample(5000, random_state=32454463)
df


# In[14]:


alt.Chart(df).mark_bar().encode(
    x = 'energy', 
    y = 'valence', 
    color = alt.Color('popularity',scale=alt.Scale(scheme='turbo')),
    tooltip = ['track_genre']
)


# Regardless of how many times it is ran, it seems songs with a lower energy and valence result in less popularity, those with a medium (around 0.5) energy and valence range between not that popular to semi-popular (20-70 range). Those with a high energy and valence range from 0-40 in popularity. 

# In[15]:


brush = alt.selection_interval(encodings=["x"], init={"x": [0,100]}) # step 1

c1 = alt.Chart(df).mark_circle().encode(
    x="energy",
    y="popularity",
    color=alt.condition(brush, "track_genre", alt.value("orchid"))
).add_selection(brush) # step 2

c2 = alt.Chart(df).mark_bar().encode(
    x="track_genre",
    y=alt.Y("count()", scale=alt.Scale(domain=[0,80])),
    color="track_genre"
).transform_filter(brush)

alt.hconcat(c1,c2) # c1|c2


# We can see from this graph that in terms of energy and popularity and when you include all the data points, pop-film does not have the most counts which would have been expected based on the 'pop_score'. This might be because we chose a wrong factor ('energy') to compare popularity with instead of a different one. 

# In[16]:


cols = ['energy', 'valence', 'danceability']


# I am now going to use OneHotEncoder to convert the 'track_genre' column into a Numpy array so we can incorporate it into LinearRegression.

# In[17]:


encoder = OneHotEncoder()


# In[18]:


encoder.fit(df[['track_genre']]) # 'track_genre' is a string column so we want to convert it into 1s and 0s


# We convert the feature names into a list so that we can later use them as new columns in df

# In[19]:


new_cols = list(encoder.get_feature_names_out()) 
new_cols


# In[20]:


df2 = df.copy() # make a copy of df to be safe


# In[21]:


df2[new_cols] = encoder.transform(df[["track_genre"]]).toarray() # transform the 'track_genre' column into array


# In[22]:


df2


# In[23]:


reg = LinearRegression(fit_intercept=False) # False so we don't include 0


# In[24]:


reg.fit(df2[cols+new_cols], df2['popularity']) # cols and new cols are out independent variables, while popularity is our dependent varaible


# In[25]:


pd.Series(reg.coef_, index=reg.feature_names_in_)


# In[26]:


pd.Series(reg.coef_, index=reg.feature_names_in_).sort_values(ascending=False, key=abs)


# From this, it seems that energy and valence are not big indicators or popularity of a song but it is danceability that determines popularity. K-pop also seems to be the most popular genre (possibly because of danceability of its songs ?)

# I am going to use KNeighborsRegressor and KNeighborsClassifier to see which of them produces the better results

# In[27]:


X_train, X_test, y_train, y_test = train_test_split(df2[cols], df2['popularity'], train_size=0.8) # split the data into a training and test set


# In[28]:


reg2 = KNeighborsRegressor(n_neighbors=10)


# In[29]:


reg2.fit(X_train, y_train)


# In[30]:


reg2.predict(X_train)


# In[31]:


mean_absolute_error(reg2.predict(X_train), y_train)


# In[32]:


mean_absolute_error(reg2.predict(X_test), y_test)


# Using mean absolute error to compare the test and training set, we see the error for the test set is greater than the training set meaning we will not be overfitting the data when using n_neighbors=10. 

# In[33]:


def get_scores(k):
    K_reg = KNeighborsRegressor(n_neighbors=k)
    K_reg.fit(X_train, y_train)
    train_error = mean_absolute_error(K_reg.predict(X_train), y_train)
    test_error = mean_absolute_error(K_reg.predict(X_test), y_test)
    return (train_error, test_error)


# We will see which k values will give us the least test error 

# In[34]:


reg_scores = pd.DataFrame({"k":range(1,150),"train_error":np.nan,"test_error":np.nan})
reg_scores


# In[35]:


for i in reg_scores.index:
    reg_scores.loc[i,["train_error","test_error"]] = get_scores(reg_scores.loc[i,"k"])


# In[36]:


reg_scores


# In[55]:


(reg_scores["test_error"]).min()


# This means n_neighbors=10 since the test error was around 18.2

# In[37]:


reg_scores["kinv"] = 1/reg_scores.k


# Since higher k values result in lower flexibility, we add a column with the reciprocal of k values. 

# In[38]:


reg_scores


# In[39]:


reg_train = alt.Chart(reg_scores).mark_line().encode(
    x = "kinv",
    y = "train_error"
)


# In[40]:


reg_test = alt.Chart(reg_scores).mark_line(color="orange").encode(
    x = "kinv",
    y = "test_error"
)


# In[41]:


reg_train+reg_test


# We can see that there is decent flexibility and variance in the beginning and all the underfitting after means there is lower flexibility.

# In[42]:


clf = KNeighborsClassifier(n_neighbors=7)


# We will now compare the result with KNeighborsClassifier

# In[43]:


clf.fit(X_train, y_train)


# In[44]:


mean_absolute_error(clf.predict(X_train), y_train)


# In[45]:


mean_absolute_error(clf.predict(X_test), y_test)


# The mean absolute error for the test set is greater than the training set, meaning we will not be overfitting for n_neighbors = 7. 

# In[46]:


def get_clf_scores(k):
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_train, y_train)
    train_error = mean_absolute_error(clf.predict(X_train), y_train)
    test_error = mean_absolute_error(clf.predict(X_test), y_test)
    return (train_error, test_error)


# In[47]:


clf_scores = pd.DataFrame({"k":range(1,150),"train_error":np.nan,"test_error":np.nan})


# In[48]:


for i in clf_scores.index:
    clf_scores.loc[i,["train_error","test_error"]] = get_clf_scores(clf_scores.loc[i,"k"])


# Process is the same as KNeighborsRegressor

# In[49]:


clf_scores


# In[56]:


clf_scores["test_error"].min()


# Using n_neighbors=7 wasn't a great choice since the test error was around 27.8 which is somewhat far

# In[50]:


clf_scores["kinv"] = 1/clf_scores.k


# In[51]:


clftrain = alt.Chart(clf_scores).mark_line().encode(
    x = "kinv",
    y = "train_error"
)


# In[57]:


clftest = alt.Chart(clf_scores).mark_line(color="orange").encode(
    x = "kinv",
    y = "test_error"
  ).properties(
      title= "Error",
       
    
)


# In[58]:


clftrain+clftest


# There is good amount of flexibility and varaince in the beginning and overfitting occurs after.

# ## Summary
# 
# From comparing KNeighborsRegressor and KNeighborsClassifier, we can see that KNeighborsRegressor would be better choice for our dataset. From the test error, we have an error of around 18% from KNeighborsRegressor so we can expect popularity to have around a 18% error as well. We can conclude there is some sort of correlation between popularity of a song with its genre. 

# ## References
# 
# Your code above should include references.  Here is some additional space for references.

# * What is the source of your dataset(s)?
# https://www.kaggle.com/code/kelvinzeng/spotify-tracks-analysis

# * List any other references that you found helpful.
# https://christopherdavisuci.github.io/UCI-Math-10-W22/Proj/StudentProjects/DanaAlbakri.html
# https://christopherdavisuci.github.io/UCI-Math-10-W22/Week6/Week6-Wednesday.html
# https://christopherdavisuci.github.io/UCI-Math-10-F22/Week7/Week7-Wednesday.html#including-a-categorical-variable-in-our-linear-regression
# https://christopherdavisuci.github.io/UCI-Math-10-F22/Week6/Week6-Friday.html#linear-regression-using-a-categorical-variable

# ## Submission
# 
# Using the Share button at the top right, **enable Comment privileges** for anyone with a link to the project. Then submit that link on Canvas.

# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=3b50ad16-4850-47bd-b8ce-5f0ca3f733e0' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>
