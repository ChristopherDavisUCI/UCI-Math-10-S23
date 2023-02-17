#!/usr/bin/env python
# coding: utf-8

# # Attempts about finding aspects that can influence the effects of Music Therapy
# 
# Author: Mingyu Chen
# 
# Course Project, UC Irvine, Math 10, F22

# ## Introduction
# 
# I was attracted by this dataset in kaggle with this link, and I realize there are a lot of perspectives that may influence the effects of music therapy. I'd love to try if I can find the connections and relationships between the music effects and those various relevant factors.
# 
# https://www.kaggle.com/datasets/catherinerasgaitis/mxmh-survey-results
# 
# To start my research, I first made some changes with the dataframe to make it more convenient for me to analyze, and then I tried to find possible factors with mainly three directions: respondents' ages and frequency of listening, the various kinds of music (with different BPM), and respondents' mental health conditions.

# ## Preparation and the processing on the dataset
# In this part, I'd love to make some changes with the dataset so it can be analyzed easily and efficiently.

# In[1]:


import pandas as pd
import numpy as np
import altair as alt
import seaborn as sns


from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree


# In[2]:


df = pd.read_csv("mxmh_survey_results.csv").dropna(axis=0)


# In[3]:


df.head()


# In this dataframe, we can find that there is a lot of data like "yes", "no" and "rarely", which are good to read but may not be so easy to process and work on. Therefore, I'd love to change them to be numbers before data analysis.

# In[4]:


df1 = df.copy()


# In[5]:


# change the format of "Timestamps"
df1["Timestamp"] = df1["Timestamp"].map(lambda x: pd.to_datetime(x))


# In[6]:


#replace "yes"/"no" with "1"/"0"
df1.replace(('Yes', 'No'), (1, 0), inplace=True) # reference, source 1


# In[7]:


# replace the words of frequency
df1.replace(('Never', 'Rarely', 'Sometimes', 'Very frequently'), (0,1,2,3), inplace=True) 


# In[8]:


df1.head()


# Check and drop if there are outliers by using Boolean Indexing:

# In[9]:


df1.shape # orginal shape of dataframe before removing the outliers


# In[10]:


# Find the first outlier with the original data
arr = df1["BPM"]
s = arr.std()
outlier = arr[np.abs(arr - arr.mean()) > 5*s]
outlier


# It's easy to realize there indeed have outliers, like the extremely large number of BPM here. We will use a while loop to repeat this code until there's no more outliers in our standard of checking.

# In[11]:


df1.index


# In[12]:


# Using a while loop to get out of the outliers.
while len(outlier) > 0:
    arr = df1["BPM"]
    s = arr.std()
    outlier = arr[np.abs(arr - arr.mean()) > 5*s]
    df1 = df1.drop(index = outlier.index);
    #repeat run this cell until len(outlier) = 0
else:
    print(f"Now the amount of outliers in df1 is {len(outlier)}")


# In[13]:


df1.shape # the shape of df1 changed, so we know we indeed dropped the outliers


# In[14]:


df1.head()


# ## Music effects in different ages with different times per day
# 
# After working on the dateset, my first attempt is to try to find the relationships between the ages and times that respondents spend to listen music every day.
# 
# I will use altair to draw some plots to help analyzing the possible relationships.

# In[15]:


df1_train, df1_test = train_test_split(df1, train_size=0.8, random_state=2)


# In[16]:


brush = alt.selection_interval()


# In[17]:


alt.Chart(df1_train).mark_circle(size=50, opacity=1).encode(
    x="Age",
    y="Hours per day",
    color=alt.Color("Music effects", scale=alt.Scale(domain=["Improve", "Worsen", "No effect"])),
    tooltip=["Age", "Hours per day","Music effects","While working"]
).properties(
    title="Music effects in different ages and times per day"
).add_selection(
    brush
)


# From this plot, we learn most respondents are aged from 10 to 40, and they usually listen to music at least 1 hour each day.
# 
# At the same time, we can intuitively be aware that the blue color which represents "improve" has a much larger amount than the others.
# 
# However, it's still hard to say there's any difference between those people who think music is helpful  and those who think music makes their conditions worsen or with no effect.

# ## Possible relationships between BPM and music effects
# 
# Because it seems to be with no obvious relationships between music effects and people's ages, I plan to try some new varibles, like the BPM.

# In[18]:


kmeans = KMeans()


# In[19]:


kmeans.fit(df1[["Hours per day", "BPM"]])


# In[20]:


df1["cluster"] = kmeans.predict(df1[["Hours per day", "BPM"]])


# In[21]:


alt.Chart(df1).mark_circle(size=50).encode(
    x="Hours per day",
    y="BPM",
    color="Music effects",
    tooltip=["Music effects","BPM"]
)


# Trying to draw the plot directly with the data of hours per day as well as the amount in the column "BPM", we can hardly find or cluster groups with people who more likely to feel music therapy to be helpful or not.
# 
# Therefore, I would like to use altair and draw the plot with automatic grouping with kmeans, and it may inlcude some new information.

# In[22]:


c1 = alt.Chart(df1).mark_circle(size=50).encode(
    x="Hours per day",
    y="BPM",
    color="cluster:N",
    tooltip="Music effects"
).add_selection(
    brush
)

c2 = alt.Chart(df1).mark_bar().encode(
    x="Music effects",
    y="BPM",
    color="cluster:N"
).transform_filter(
    brush
)

c1|c2


# Unfortunately, similar with the result before, the plot shows little connections between the BPM and the music effects.
# 
# To be more specific, the clustering ways show some relationships among groups, but that cannot help us understand if there's a influence factor that leads to different music effects.

# ## Respondents' mental states and music effects
# 
# We failed to find the differences in BPM between people who feel music therapy is helpful and not helpful. Actually, music effects show totally independent with all of the variables we mentioned.
# 
# In this case, I will try the respondents' self feelings about their mental health, which includes anxiety level, depression level, insomnia level, and OCD.

# In[23]:


df1.columns


# In[24]:


df2 = df1[['BPM','Anxiety', 'Depression', 'Insomnia', 'OCD', 'Music effects']]
df2


# By creating a specific df2, I'm going to find their relationships among each other.
# 
# Using Pairplot may be a good attempt, and I hope we can find some clusters with distinct "colors"

# In[25]:


sns.pairplot(df2, hue="Music effects")


# We are able to see that there're a lot of orange points than the other two colors' points, but it's hard to say if there's any influence came from the four variables(Anxiety, Depression, Insomnia, and OCD)

# In[26]:


clf = DecisionTreeClassifier(max_leaf_nodes=10, random_state=1)


# In[27]:


df2_train, df2_test = train_test_split(df2, train_size=0.9, random_state=2)


# In[28]:


df2_train.head()


# In[29]:


cols = ["BPM", "Anxiety", "Depression", "Insomnia", "OCD"]


# In[30]:


clf.fit(df2_train[cols], df2_train["Music effects"])


# In[31]:


fig = plt.figure(figsize=(10, 5)) # reference, source 2
_ = plot_tree(clf, 
                   feature_names=clf.feature_names_in_,
                   class_names=clf.classes_,
                   filled=True)


# In this DecisionTreeClassifier plot, almost all the variables can lead the results, and they are arranged in a uncertain way with no specific differences. At this time, I believe these conditions of mental health cannot lead to differences in music effects as well.

# ## Summary
# 
# In this project, I tried to analyze mainly three kinds of data: Age and frequency, BPM of music, and respondents' mental health condition. Unfortunately, all of these three fields of variables show to have only little relationship with final music effects.
# 
# In the case that we don't know what's the most influential factor that can lead to different results of music therapy, it's lucky to know that most of the results show music therapy is helpful. I hope we will be able to find out the most important factor one day.

# ## References
# 
# Your code above should include references.  Here is some additional space for references.

# * What is the source of your dataset(s)?

# Source 1: Turn yes/no to be 1/0
# https://stackoverflow.com/questions/40901770/is-there-a-simple-way-to-change-a-column-of-yes-no-to-1-0-in-a-pandas-dataframe
# 
# Source 2: Change the size of plot trees
# https://stackoverflow.com/questions/59447378/sklearn-plot-tree-plot-is-too-small

# * List any other references that you found helpful.

# Detect and exclude outliers
# https://stackoverflow.com/questions/23199796/detect-and-exclude-outliers-in-a-pandas-dataframe

# ## Submission
# 
# Using the Share button at the top right, **enable Comment privileges** for anyone with a link to the project. Then submit that link on Canvas.

# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=c0b5f455-d874-471d-a8c7-a32d5a546228' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>
