#!/usr/bin/env python
# coding: utf-8

# # Video Game Sales
# 
# Author: Aditya Krishnan Radhakrishnan
# 
# Course Project, UC Irvine, Math 10, F22

# ## Introduction
# 
# The goal of this project is to analyze what factors affect the sales of popular video games in the past couple decades. The project also aims to explore regional preferences of games and create a model to predict the release platform of a game. This analysis will incorporate data manipulation using the Pandas library, plotting graphs using Altair and some machine learning algorithms.

# In[1]:


import pandas as pd
import altair as alt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression


# ## Exploring and cleaning the data

# In[2]:


df = pd.read_csv("vgsales.csv")
df.dropna(axis=0, inplace=True)
df.info()


# In[3]:


df.head()


# There are still some games in the dataset with "Unknown" publishers so we will use boolean indexing to delete those rows.

# In[4]:


df = df[df["Publisher"] != 'Unknown'].copy()


# The 'Ranked' column is quite redundant when we already have index.

# In[5]:


df.drop(['Rank'], axis=1, inplace=True)


# The 'Year' column should be converted to datetime values rather than floats.

# In[6]:


df['Year'] = df['Year'].map(lambda x: int(x)).copy()


# In[7]:


df['Year'] = df['Year'].map(lambda x: f"{x}-01-01").copy()


# In[8]:


df['Year'] = pd.to_datetime(df['Year']).copy()


# Our dataset is still quite large. We can shrink it by removing the rows closer to the end as those games are not so popular anyways.

# In[9]:


df = df[:5000].copy()
df


# We are also changing the genre "Platform" to "Platformer" for the future.

# In[10]:


df['Genre'] = df['Genre'].map(lambda x: "Platformer" if x == "Platform" else x).copy()


# Now we will graph the sales of the video games over time, grouped by genre. From this graph, we can tell what genres are dominated by different companies.

# In[11]:


sel = alt.selection_single(fields=['Genre'], bind="legend")

c1 = alt.Chart(df).mark_circle().encode(
    x='Year',
    y='Global_Sales',
    color='Genre:N',
    tooltip=['Name', 'Platform', 'Publisher']
).add_selection(sel)

c2 = alt.Chart(df).mark_bar().encode(
    x=alt.X('Publisher', sort='-y'),
    y='sum(Global_Sales)'
).transform_filter(sel)

c1|c2


# ## Japan's Preferences (Regression)
# We will use scikit-learn's LinearRegression to figure out what factors affect the popularity of video games in Japan. We will consider the year, top 5 publishers, and genres. Since we are using regression (not classification), we must first make sure the variables we are using are numerical and not categorical. We will use OneHotEncoder for this.
# Also, we will only be considering the top 5 publishers to not make the data too complex.

# In[12]:


df2 = df[(df['Publisher'] == 'Nintendo') | (df['Publisher'] == 'Electronic Arts') | (df['Publisher'] == 'Activision') | (df['Publisher'] == 'Sony Computer Entertainment') | (df['Publisher'] == 'Ubisoft')].copy()


# We convert the release years from datetime values to int values.

# In[13]:


df2['Year2'] = df2['Year'].dt.year


# We make series of boolean values for the top 5 publishers using OneHotEncoder.

# In[14]:


encoder_pub = OneHotEncoder()


# In[15]:


encoder_pub.fit(df2[["Publisher"]])


# In[16]:


arr_pub = encoder_pub.transform(df2[["Publisher"]]).toarray()
arr_pub


# In[17]:


pub_list = [encoder_pub.get_feature_names_out()[i][10:] for i in range(5)]
pub_list


# In[18]:


df2[pub_list] = arr_pub
df2


# We then repeat the same process for the genres.

# In[19]:


encoder_g = OneHotEncoder()


# In[20]:


encoder_g.fit(df2[["Genre"]])


# In[21]:


arr_g = encoder_g.transform(df2[["Genre"]]).toarray()
arr_g


# In[22]:


g_list = [encoder_g.get_feature_names_out()[i][6:] for i in range(12)]
g_list


# In[23]:


df2[g_list] = arr_g
df2


# Below is a list of the columns we want to use for the regressor.

# In[24]:


cols = list(df2.columns)[10:]
cols


# We now instantiate the regressor object, fit the data, and get the coefficients for the predicted values.

# In[25]:


reg = LinearRegression(fit_intercept=False)


# In[26]:


reg.fit(df2[cols], df2['JP_Sales'])


# In[27]:


df2['Pred'] = reg.predict(df2[cols])


# In[28]:


pd.Series(reg.coef_, index=cols)


# We can see from the coefficients that the coefficient for year is negative; meaning video games got less popular over time in Japan. We can also see that Japanese sales are higher for Nintendo games. In terms of genre, we can see that role-playing games are popular in Japan (likely due to the immense success of Pokemon).
# 
# We can graph the prediction line on top of the data.

# In[29]:


sel2 = alt.selection_single(fields=["Publisher"], bind="legend", empty="none")

c3 = alt.Chart(df2).mark_circle().encode(
    x='Year',
    y='JP_Sales',
    color='Publisher',
    tooltip = ["Name", "Publisher", "Genre"],
    opacity=alt.condition(sel2, alt.value(1), alt.value(0.2)),
    size=alt.condition(sel2, alt.value(60), alt.value(20)),
).add_selection(sel2)

c4 = alt.Chart(df2).mark_line(color='black').encode(
    x='Year',
    y='Pred'
).transform_filter(
    sel2
)

c3+c4


# We can see from the graph that Nintendo is dominating the Japanese video game market (along with Sony to a certain extent) while other American publishers like Electronic Arts and Activision are not doing as well.

# ## Platforms used by publishers (Classification)
# 
# We will use the K-Nearest Neighbors Classifier to predict the platforms used by the top 5 publishers. Other data we will use to help train the model includes the sales, genre and release year.

# In[30]:


X = df2[['Activision',
 'Electronic Arts',
 'Nintendo',
 'Sony Computer Entertainment',
 'Ubisoft','NA_Sales',
 'EU_Sales',
 'JP_Sales',
 'Other_Sales',
 'Global_Sales',
 'Action',
 'Adventure',
 'Fighting',
 'Misc',
 'Platformer',
 'Puzzle',
 'Racing',
 'Role-Playing',
 'Shooter',
 'Simulation',
 'Sports',
 'Strategy',
 'Year2']]
y = df2["Platform"]


# In[31]:


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=2)


# Similar to before, we will instantiate the classifier object, fit the data, and obtain the scores.

# In[32]:


clf = KNeighborsClassifier(n_neighbors=30)


# In[33]:


clf.fit(X_train, y_train)


# In[34]:


clf.score(X_train, y_train)


# In[35]:


clf.score(X_test, y_test)


# From the classifier's score on the training set and the test set, we can see that they are fairly close (within 5%). From this we know that we are not overfitting the data with our parameters as the training score is not significantly better than the test score. Although we can get a higher accuracy on the training data by using less neighbors, we will be sacrificing accuracy on the test set which is much more important because it evaluates how well our model can perform on more data.
# 
# Looking at the actual values however, we can see that the model is not very accurate, with an accuracy of only 44%. Thus, we can say that it is hard to predict the platform of a game from sales in different regions, the publishers of the game, and its genre.

# ## Summary
# 
# To summarize, we analyzed the sales of video games over time using data on publishers, genres and platforms and cleaned and explored the data using Pandas and Altair respectively. We used OneHotEncoder to allow us to use categorical variables in scikitlearn's linear regression and reached conclusions on Japan's taste in video games. We then used a KNeighbors Classifier to try and predict the platform a game was released on, but ended up with a not-so-accurate model meaning it is difficult to predict the platform.

# ## References
# 
# Your code above should include references.  Here is some additional space for references.
# 
# The sorted bar graph was adapted from https://altair-viz.github.io/gallery/bar_chart_sorted.html
# Using if and else in lambda functions from https://thispointer.com/python-how-to-use-if-else-elif-in-lambda-functions/

# * What is the source of your dataset(s)?
# This dataset called "Video Game Sales" was taken from Kaggle and was uploaded by GREGORYSMITH. https://www.kaggle.com/datasets/gregorut/videogamesales?resource=download

# * List any other references that you found helpful.

# ## Submission
# 
# Using the Share button at the top right, **enable Comment privileges** for anyone with a link to the project. Then submit that link on Canvas.

# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=c1fbf3b2-71a7-403c-845c-8b3c8147564c' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>
