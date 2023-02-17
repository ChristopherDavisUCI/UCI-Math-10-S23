#!/usr/bin/env python
# coding: utf-8

# # Hollywood Movie Gross Income
# 
# Author: Nicholas Bogarin
# 
# Course Project, UC Irvine, Math 10, F22

# ## Introduction
# 
#  My project is going to go in depth towards a dataset that describes the Highest Hollywood Grossing Movies. I will be mainly reorganzing the dataset in order for it to be understood more before computating codes that can help predict questions that help uncover the reasons for the highest income sale and more. It will be a more general idea on which movies are the highest sold not only in the domestic area that it was created and released in, but the overall impact on it to the world sales. Being able to identify which productor created the most popular ones and what is predicted to come to his next future releases :) 

# ### Import Library:

# In[1]:


import numpy as np
import pandas as pd
import altair as alt
import plotly.express as px

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# ## Cleansing of the dataset
# 
# In this section, I will not only be cleansing part of the dataset, but I will step by step create sub data frames that help us furhter analyse the datasets together to grasp an idea on what movies are set out to be popular hits financially and physically. This will dive into the main part of the project where I will use numerous of topics used in Pandas to make it more efficiently for more machine learning in the next section :)

# ### Highest Holywood Grossing Movies dataset

# In[2]:


df_movies = pd.read_csv("/work/archive (6)/Highest Holywood Grossing Movies.csv")
df_movies


# In[3]:


df_movies.shape


# - This dataset is from kaggle and is a set that contains numerous of the Highest Hollywood Grossing Movies that have occured for the past century. It has 918 rows and 11 columns that describe the movies and information about the movie like the info, distributor, release date, the genre, length, license, and the overall Sales that come within domestic, international, and world. As we can see from the dataset there are values that are missing or a column that is not important in the set, so we are going to remove them to help make our dataset cleaner and shorter(I will not be removing the missing row values here first because later on it will be cleaner if I do it during that time) :)

# In[4]:


df_movies = df_movies.drop({'Unnamed: 0','Movie Info'}, axis = 1)


# In[5]:


df_movies


#    Now we can see that the columns are decreased to 9 columns as we removed the unnecessary index column and the movie information since we would not be able to graph them or have any use of them. Now we see in the Genre column how all the genre types are combined in a list form and it would have been better if we split the genre string in order to be able to categorize them into genres to get a deeper understanding of which genre is set to be more popular in sales :)

# In[6]:


df_movies["Genre"] = df_movies["Genre"].astype(str)


# In[7]:


df_movies['Genre']


# In[8]:


df_genres =pd.DataFrame(df_movies["Genre"].str.split(',', expand=True).values,
             columns=['Genre1', 'Genre2','Genre3','Genre4','Genre5','Genre6', 'Genre7','Genre8'])
df_genres


# In[9]:


df_genres = df_genres.fillna("")


# Now that we created a dataframe that seperates all the genres that were described in each movie into seperate columns we are going to combine it with the movies dataset in order to graph them together in this unified code. We see that there are numerous of columns that do not contain values for them in example say Genre5 or 7, so we are mainly going to focus only on Genre 1 since each movie atleast contains 1 genre type. Now we also remove the all the rows that contain missing values in the movie dataframe to not have any missing values :)

# In[10]:


df_movies1 = pd.concat([df_movies,df_genres], axis= 1)
df_movies1


# In[11]:


df_movies1 = df_movies1.dropna(axis = 0)
df_movies1 = df_movies1.drop('Genre', axis = 1)
df_movies1


# In[12]:


df_movies1["Genre1"].value_counts()


# In[13]:


df_movies1["Genre1"] = df_movies1["Genre1"].replace(["['Action'","['Adventure'","['Comedy'","['Drama'","['Comedy']","['Biography'","['Crime'","['Horror'","['Drama']","['Fantasy'","['Animation'","['Mystery'","['Horror']","['Documentary'"],["Action","Adventure","Comedy","Drama","Comedy","Biography","Crime","Horror","Drama","Fantasy","Animation","Mystery","Horror","Documentary"])


# In[14]:


df_movies1["Genre1"].value_counts()


# In[15]:


df_movies1 = df_movies1.drop(['Genre2','Genre3','Genre4','Genre5','Genre6','Genre7','Genre8'], axis = 1)


# In[16]:


df_movies1.shape


# Now with this new shortened sub data frame we can dive into the logistics and visualizations to learn about each film and what they are related to towards distributors and sale earnings :)

# ## Graphing the Dataset
# ###  Visualizing which Distributor will will have the highest World Sale Earnings

# Here we are first going to graph using altair and compare 3 graphs that demonstrate each distributor of the movies and find the mean value for all the Domestic, International, and World sales in $. After identifying which Distributor of all contains the highest gross income within the movies, we will predict their world sale earnings for all movies they have produced to determine future results :)

# In[17]:


c = alt.Chart(df_movies1).mark_bar().encode(
    x= alt.X("Distributor", scale=alt.Scale(zero=False)),
    y="mean(Domestic Sales (in $))",
    color=alt.Color("License", scale=alt.Scale(scheme="redpurple")),
    tooltip = ['Title', "Movie Runtime"]
)


# In[18]:


c1 = alt.Chart(df_movies1).mark_bar().encode(
    x= alt.X("Distributor", scale=alt.Scale(zero=False)),
    y="mean(International Sales (in $))",
    color=alt.Color("License", scale=alt.Scale(scheme="redpurple")),
    tooltip = ['Title', "Movie Runtime"]
)


# In[19]:


c2 =alt.Chart(df_movies1).mark_bar().encode(
    x= alt.X("Distributor", scale=alt.Scale(zero=False)),
    y="mean(World Sales (in $))",
    color=alt.Color("License", scale=alt.Scale(scheme="redpurple")),
    tooltip = ['Title',"Movie Runtime"]
)


# In[20]:


alt.hconcat(c,c1,c2)


# As we see on the graphs, there is numerous of high values for 3 or more distributors and there is a huge gap bwtween License preferences as pg 13 contains higher sales than any other rated movie. there are numerous of graphs that goes by each sale seperately and it can be hard to visualize and identify the highests increasing. So I incorporated my extra component with this section using plotly to help us identifty the values in a 3D form :)

# In[21]:


import plotly.express as px
fig = px.scatter_3d(df_movies1, x='Domestic Sales (in $)', y='International Sales (in $)', z='World Sales (in $)',
              symbol='Genre1', color = "Distributor")
fig.show()


# After seeing this diagram, we can go further in depth and I want to explore the World Sales since it is the highest value in income within each movie :)

# In[22]:


fig = px.bar(df_movies1, x="Genre1", y="World Sales (in $)", color="Distributor", title="Genre 1")
fig.show()


# In[23]:


fig = px.sunburst(df_movies1, path=['Distributor', 'Genre1',"Title"], values='World Sales (in $)', color= "World Sales (in $)")
fig.show()


# From both of the graphs, the bar and sunburt, we can see that the most popular and most created genre within this dataset is shown to be Action and it contains almost all of the highest movie films that gained a huge amount of gross. What is more important from these graphs is we can see that Walt Disney is the majority of the values that gained high gross income, so we will explore more in depth towards just the ditributor Disney and predict their outcomes in the films :)

# ## Disney Distributor World Sales 

# In[24]:


df_Disney = df_movies1[df_movies1["Distributor"] == "Walt Disney Studios Motion Pictures"].copy()
df_Disney


# Here I incorporated a student's project example to help me predict the values and World Sales for future movies on Disney films using linear regression :)

# In[25]:


cols = [c for c in df_Disney.columns if is_numeric_dtype(df_Disney[c])]


# In[26]:


df_Disney["Release Date"]= pd.to_datetime(df_Disney["Release Date"]).astype(int)


# In[27]:


reg = LinearRegression()
cols=['Domestic Sales (in $)','International Sales (in $)','World Sales (in $)']
reg.fit(df_Disney[cols],df_Disney["World Sales (in $)"])
pd.Series(reg.coef_,index=cols)


# In[28]:


reg.coef_


# Here we see the coefficent for the prediction on all 3 types of sales and we are going ot incorporate it into out graph and only use the Domestic and World Sales to demonstrate the difference and importance on World Sales that help increase numerous of the film's incomes :)

# In[29]:


df_Disney["Pred"] = reg.predict(df_Disney[cols])


# In[30]:


c_train,c_test,d_train,d_test = train_test_split(df_Disney[cols],df_Disney["World Sales (in $)"],test_size = 0.5, random_state=10)
reg.fit(c_train,d_train)
pred= reg.predict(c_train)
reg.score(c_train, d_train)


# In[31]:


corr = df_Disney.corr()
corr.sort_values(["World Sales (in $)"], ascending = False, inplace = True)
print(corr.Pred)


# ### Prediction Graph 

# In[32]:


sel = alt.selection_single(fields=["Genre1"], empty="none")

base = alt.Chart(df_Disney).mark_circle().encode(
    x="Domestic Sales (in $)",
    y="World Sales (in $)",
    tooltip=["Title", "Movie Runtime"],
    size=alt.condition(sel, alt.value(80),alt.value(20)),
    color=alt.Color("Genre1", scale=alt.Scale(scheme="Paired")),
    opacity=alt.condition(sel, alt.value(1), alt.value(0.5))
).add_selection(sel)

text = alt.Chart(df_Disney).mark_text(y=20, size=20).encode(
    text="Genre1",
    opacity=alt.condition(sel, alt.value(1), alt.value(0))
)

n1=alt.Chart(df_Disney).mark_line(color="lightgray").encode(
    x='Domestic Sales (in $)',
    y='Pred',
)
n1

n = base+text
n+n1


# In[33]:


fig = px.sunburst(df_Disney, path=['Genre1',"Title"], values='World Sales (in $)', color= "World Sales (in $)")
fig.show()


# ## Summary
# 
# Overall, I was able to find a dataset on kaggle that gave us information about Hollywood's Highest Grossing Movies and go into depth on the logistics of what makes a movie popular and have the highest gross sales. If it is simple as choosing a genre, or making the movie pg 13 rather than pg, or as complex as choosing the distributor to help boost up credibility, all these we were able to identify in order to grasp an understanding on basic movie preferences. We were also able to predict the future values that are created to find the domestic and world sales of Walt Disney's Movie films and more. 

# ## References
# 
# Your code above should include references.  Here is some additional space for references.

# Ploty 3D Scatter Plot :[https://plotly.com/python/3d-scatter-plots/]

# Plotly Bar Chart: [https://plotly.com/python/bar-charts/]

# Plotly Sun Chart: [https://plotly.com/python/sunburst-charts/]

# * What is the source of your dataset(s)?

# Kaggle: [https://www.kaggle.com/code/yasirnikozaiofficial/highest-grossing-movies-of-hollywood-eda/data]

# * List any other references that you found helpful.

# Student Example: [https://christopherdavisuci.github.io/UCI-Math-10-S22/Proj/StudentProjects/KehanLi.html]

# ## Submission
# 
# Using the Share button at the top right, **enable Comment privileges** for anyone with a link to the project. Then submit that link on Canvas.

# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=528e2efc-1263-48b4-ba00-f117f107b0d8' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>
