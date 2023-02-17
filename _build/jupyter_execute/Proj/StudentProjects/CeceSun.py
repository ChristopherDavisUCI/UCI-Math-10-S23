#!/usr/bin/env python
# coding: utf-8

# # What contributes to a movie's commercial success?
# 
# Author: Cece Sun
# 
# Course Project, UC Irvine, Math 10, F22

# ## Introduction
# 
# This project investigates certain features that movies screening in the theater might have and dives into the correlation between each feature and the commercial success of a movie (i.e. the revenue each movie makes). Meanwhile, this project also shows us what the top revenue/top popularity movies and the top directors would be.
# One important factor of a success of a movie is how much revenue it can generate. How is this feature correlating with others such as a movie's budget, popularity, and ratings? Let's take a look together.

# ## Getting Ready
# 
# Cleaning data and preparing it for future analysis and operations. 

# In[1]:


import pandas as pd


# There are 2 datasets used in this project. Merge them together by each movie's id number.

# In[2]:


df_c = pd.read_csv("tmdb_5000_credits.csv")
df_c.head(3)


# In[3]:


df_m = pd.read_csv("tmdb_5000_movies.csv")
df_m.head(3)


# In[4]:


df_m=df_m.rename(columns={"id" : "movie_id"})


# In[5]:


df = df_m.merge(df_c, on="movie_id")
df.head(3)


# In[6]:


df.columns


# Get rid of the overlapped columns and the data we are not interested in.

# In[7]:


df2 = df.drop(columns = ["movie_id","homepage", "title_x", "title_y", "overview", "status", "tagline","spoken_languages"])
df2.head(3)


# Get rid of all the na values.

# In[8]:


df3 = df2.dropna()
df3.head(3)


# Rename some column names to simplify future codes.

# In[9]:


df4=df3.rename(columns = {"original_language":"language", "original_title":"title", "production_companies":"companies", "production_countries":"countries", "release_date":"date"})


# Preparing the categorical data for future uses.
# Particularly, extract the director name for each movie from the "crew" column.

# In[10]:


from ast import literal_eval
features = ["genres", "keywords", "companies", "countries", "cast", "crew"]
for feature in features:
    df4[feature] = df4[feature].apply(literal_eval)


# In[11]:


import numpy as np


# In[12]:


def get_director(x):
    for i in x:
        if i["job"] == "Director":
            return i["name"]
    return np.nan 


# In[13]:


def get_list(x):
    if isinstance(x, list):
        names = [i["name"] for i in x]
        if len(names)>3:
            names = names[:3]
        return names
    return[]


# In[14]:


df4["director"] = df4["crew"].apply(get_director)

features = ["cast", "keywords", "genres","companies", "countries"]
for feature in features:
    df4[feature] = df4[feature].apply(get_list)


# In[15]:


df5 = df4.drop(columns = ["crew"]).dropna()
df5.head(3)


# In[16]:


df5["date"] = pd.to_datetime(df5["date"])


# Reorder the columns' positions in a way I want them to be.

# In[17]:


ordered_col = ["title", "date", "revenue","budget", "popularity","runtime", "vote_average", "vote_count", "countries", "language", "genres", "keywords", "director", "cast"]


# In[18]:


df6=df5[ordered_col]


# ## Movie Ranking
# 
# What are the top5 movies based on their revenue? Popularity? `##`

# In[19]:


df6.sort_values(by="revenue", ascending=False).head(5)


# In[20]:


pop = df6.sort_values('revenue', ascending=False)
import matplotlib.pyplot as plt
plt.figure(figsize=(12,4))

plt.barh(pop['title'].head(8),pop['revenue'].head(8), align='center',
        color='indianred')
plt.gca().invert_yaxis()
plt.xlabel("Revenue")
plt.title("Profitable Movies")


# From this bar chart we can see that "Avatar", "Titanic", and "The Avegers(the first one)" are the top 3 most profitable movies.

# In[21]:


df6.sort_values(by="popularity", ascending=False).head(5)


# In[22]:


pop2 = df6.sort_values('popularity', ascending=False)
import matplotlib.pyplot as plt
plt.figure(figsize=(12,4))

plt.barh(pop2['title'].head(8),pop2['popularity'].head(8), align='center',
        color='skyblue')
plt.gca().invert_yaxis()
plt.xlabel("Popularity")
plt.title("Popular Movies")


# From this bar chart we can see that "Minions", "Interstellar", and "Deadpool" are the top 3 most popular movies, which diverges from the top 3 most profitable movies.

# ## Computing a movie_score for each movie
# 
# ### What are IMDb movie ratings?
# IMDb registered users can cast a vote (from 1 to 10) on every released title in the database. Individual votes are then aggregated and summarized as a single IMDb rating, visible on the title's main page. By "released title" we mean that the movie (or TV show) must have been shown publicly at least once (including festival screening).
# 
# Users can update their votes as often as they'd like, but any new vote on the same title will overwrite the previous one, so it is one vote per title per user.
# 
# 
# ### How are the ratings calculated?
# They take all the individual ratings cast by IMDb registered users and use them to calculate a single rating. They don't use the arithmetic mean (i.e. the sum of all votes divided by the number of votes), although they do display the mean and average votes on the votes breakdown page; instead the rating displayed on a title's page is a weighted average. 
# 
# The formula for calculating the Top Rated 250 Titles gives a true Bayesian estimate:
# weighted rating (WR) = (v ÷ (v+m)) × R + (m ÷ (v+m)) × C where:
# 
# R = average for the movie (mean) = (Rating)
# v = number of votes for the movie = (votes)
# m = minimum votes required to be listed in the Top 250 (currently 25000)
# C = the mean vote across the whole report (currently 7.0) `##`

# In[23]:


C = df6["vote_average"].mean()
C


# In[24]:


m = df6["vote_count"].quantile(0.9)
m


# In[25]:


movies2 = df6.copy().loc[df6["vote_count"] >= m]
len(movies2)


# In[26]:


def weighted_rating(x, m=m, C=C):
    v = x["vote_count"]
    R = x["vote_average"]
    return(v/(v+m)*R) + (m/(m+v)*R)


# In[27]:


movies2["score"] = movies2.apply(weighted_rating, axis=1)
df6["movie_scores"] = movies2["score"].round(2)


# In[28]:


df6.sort_values(by="movie_scores", ascending=False).head(5)


# ## Visualize movies' revenue, budget, and ratings
# 
# Using altair to display charts containing movies' revenue, budget, and ratings information 

# In[29]:


import altair as alt


# In[30]:


df9 = df6[df6["revenue"] != 0]
df9


# In[78]:


alt.Chart(df9).mark_circle().encode(
    x="budget",
    y="revenue",
    color="language",
    tooltip=("title", "budget", "revenue")
)


# From this chart we can learn that generally the higher the budget, the higher the revenue. However, the revenue of a movie can depend on other features as well (not entirely affected by the budget), as we can see the highest budget movie "Pirates of the Caribbean: On Stranger Tides" is not even in the top10 most profitable movies. And "Avatar", which generated the highest revenue, did not spend the most money for production.

# In[32]:


alt.Chart(df9).mark_circle().encode(
    alt.X("movie_scores",
        scale=alt.Scale(zero=False)
    ),
    y="revenue",
    size="popularity",
    color="title",
    tooltip=("title", "revenue", "popularity", "movie_scores")
)


# From this chart, we can see that a movie's rating does not necessarily correlates its commercial success (revenue). Most of the top profitable movies distribute in the mid range of the movie ratings (6.0~8.0).

# ## Who are the top directors?
# 
# Making a sub-dataframe containing the top100 most profitable movies using their "revenue" ranking. 
# Look for the top3 directors in this sub-dataframe by counting how many top100 movies are directed by them. 
# Draw charts for all the movies made by these top3 directors separately.

# In[33]:


df_top100=df6.sort_values("revenue", ascending=False).head(100)
df_top100["director"].value_counts().sort_values(ascending=False).head(3)


# In[34]:


df_topDirectors = df6.loc[(df6["director"] == "Peter Jackson") | (df6["director"] == "Michael Bay") | (df6["director"] == "Christopher Nolan")]


# In[35]:


alt.Chart(df_topDirectors).mark_circle().encode(
    x=alt.X("movie_scores", scale=alt.Scale(zero=False)),
    y="revenue",
    size="popularity",
    color="title:N",
    tooltip=("director", "title", "revenue", "movie_scores")
).facet("director").resolve_scale(
    x='independent'
)


# From these charts we can clearly see the comparison of these three directors. All three of them have directed some high office box movies. Michael Bay's movies have the lowest range of movie ratings among the three. And Nolan has directed the most popular movie "Insterstellar" among all the movies directed by these three directors.

# ## Relationship between a movie's revenue and its bugdet
# 
# Using linear regression to plot the regression line for "revenue" and "budget".

# In[36]:


from sklearn.linear_model import LinearRegression


# Create and fit the model

# In[37]:


reg = LinearRegression()
reg.fit(df6[["budget"]], df6["revenue"])


# Making Predictions

# In[38]:


df6["pred"] = reg.predict(df6[["budget"]])


# In[39]:


base = alt.Chart(df6).mark_circle().encode(
    x="budget",
    y="revenue"
)
base


# In[40]:


c1 = alt.Chart(df6).mark_line().encode(
    x="budget",
    y="pred"
)
base+c1


# In[41]:


reg.intercept_


# In[42]:


reg.coef_


# ## Coefficients of "popularity" and "runtime"
# 
# Using linear regression and feature_names_in_ to get the coefficients of feature "popularity" and "runtime".

# In[43]:


cols2 = ["popularity", "runtime"]


# In[44]:


reg2 = LinearRegression()
reg2.fit(df6[cols2], df6["revenue"])


# In[45]:


pd.Series(reg2.coef_, index=reg2.feature_names_in_)


# ## Which genres is more likely to bring a movie high revenue?
# 
# 
# Use One hot encoding to convert the categorical data variables to be provided to machine and deep learning algorithms and compute the contribution that each genres makes to the movie's success.

# Each movie might be classified into more than one genres. We use the first genres in each movie's "genres" feature as the one that can describe the movie the most.

# In[46]:


g2 = [c[:1] for c in df6["genres"].tolist()]


# In[47]:


mystring = [str(c) for c in g2]


# In[48]:


df6["g2"] = mystring


# In[49]:


from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
encoder.fit(df6[["g2"]])


# In[50]:


new_cols = list(encoder.get_feature_names_out())


# In[51]:


df7 = df6.copy()
df7[new_cols] = encoder.transform(df6[["g2"]]).toarray()


# In[52]:


encoder.fit_transform(df6[["g2"]])


# In[53]:


encoder.fit_transform(df6[["g2"]]).toarray()


# In[54]:


reg4 = LinearRegression(fit_intercept = False)
reg4.fit(df7[new_cols], df7["revenue"])


# In[55]:


pd.Series(reg4.coef_, index = reg4.feature_names_in_).sort_values(ascending=False)


# ## Use a movie's budget, popularity, revenue, rating, and runtime informaton to predict its language.
# 
# Using train test split to split the data into train and test parts. 
# Make budget, popularity, revenue, rating, and runtime the input variables in order to predict its language.
# Use DecisionTreeClassifier and matplotlib to calculate the train and test score (to determine if it's overfitting) and display the classification tree.

# In[56]:


from sklearn.model_selection import train_test_split


# In[57]:


df8 = df6.dropna()


# In[58]:


len(df8["language"].unique())


# In[59]:


input_cols = ["budget", "popularity", "revenue", "movie_scores", "runtime"]


# In[60]:


X = df8[input_cols]
y = df8["language"]


# In[61]:


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, random_state=0)


# In[62]:


from sklearn.tree import DecisionTreeClassifier


# In[63]:


clf = DecisionTreeClassifier(max_leaf_nodes=6)


# In[64]:


clf.fit(X_train, y_train)


# In[65]:


clf.score(X_train, y_train)


# In[66]:


clf.score(X_test, y_test)


# The test score is lower than the train score but not too much lower, suggesting that the model is not overfitting.

# In[67]:


import matplotlib.pyplot as plt
from sklearn.tree import plot_tree


# In[68]:


fig = plt.figure()
_ = plot_tree(clf,
                    feature_names = clf.feature_names_in_,
                    class_names = clf.classes_,
                    filled = True)


# ## What words appear in movies' titles most frequently?
# 
# Use wordcloud and plt to display an image containing the most frequent words appearing in movies' title. 
# The bigger the word, the more frequently it appears.

# In[69]:


from wordcloud import WordCloud

plt.figure(figsize = (12, 12))
token_title = ' '.join(df8['title'].values) 

wordcloud = WordCloud(max_font_size=None, background_color='white', width=1200, height=1200).generate(token_title)
plt.imshow(wordcloud)
plt.title('Top words from movie titles ')
plt.axis("off") 
plt.show()


# ## Summary
# 
# Throughout this project, I investigate several important features of a movie, and find out their contribution upon whether the movie is a commercial success. It turns out that a high popularity movie is more likely to generate good revenue than a highly-rated movie. I also show the audience some interesting ranking of the movies.

# ## References
# 
# Your code above should include references.  Here is some additional space for references.

# * What is the source of your dataset(s)?
# 
# https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

# * List any other references that you found helpful.

# https://help.imdb.com/article/imdb/track-movies-tv/ratings-faq/G67Y87TFYYP6TWAV#
# https://medium.com/analytics-vidhya/how-to-use-machine-learning-approach-to-predict-movie-box-office-revenue-success-e2e688669972
# 
# https://www.analyticsvidhya.com/blog/2021/05/how-to-build-word-cloud-in-python/
# https://www.kaggle.com/code/ibtesama/getting-started-with-a-movie-recommendation-system

# ## Submission
# 
# Using the Share button at the top right, **enable Comment privileges** for anyone with a link to the project. Then submit that link on Canvas.

# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=bd32a67a-e80b-4a4d-b34c-dc59f2edbbf6' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>
