#!/usr/bin/env python
# coding: utf-8

# # Chip from Specs
# Author: **Eric Jones** 
# Email: ericjones8612@gmail.com
# 
# Course Project, UC Irvine, Math 10, F22

# ## Introduction
# In this notebook we will be taking a look at various memory chips of Graphics Processing Units, GPU's. In particular, we will be analyzing the specs of the memory along with the manufacturer and bus of the card in order to use machine learning to attempt to determine the type of memory.

# 
# ## Importing Libraries and Checking Out the Data

# In[1]:


# Libaries
import pandas as pd
from pandas.api.types import is_numeric_dtype
import altair as alt
import numpy as np
import random as rng
import plotly.express as px
# Machine Learning
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


# First lets import the data set using the pandas read function. Let's quickly glimpse at the columns and some random rows. Some things to note are the abundance of missing values, we won't immediately throw these rows out as some missing values are in columns we won't be needing.
# 
# Let's focus on the bottom two rows we can currently see. If you scroll the Dataframe all the way over to the right you can see both cards share the same memory type (memType), that being DDR (DDR1). However, if we scroll back to the left we notice that despite having the same RAM type the memory is not clocked the same (memClock), and the one that is clocked faster has a significantly larger bus width (memBusWidth)

# In[2]:


df_pre = pd.read_csv('gpu.csv')
df_pre.sample(5, random_state=16788682)


# This leads us to my question. **Can we accurately guess the type of memory a graphics card has based off the memory's attributes?** As we saw before, the same type of RAM can be clocked and built totally differently, but surely there must be some correlation.

# ## Preping the data

# Earlier I mentioned that there were an abundance of missing values. These are hard to work with, so we will do our best to drop any row with missing values. Although as it turns out, we cannot just use a simple .dropna() method on the entire Dataframe as we would be left with a Dataframe with zero rows. So first we must 'trim the fat' so to speak, by making a new Dataframe with only the interesting columns.
# 
# First we will make a list with every column with the string 'mem' in it, as these will be the most relevant.

# In[3]:


memcols = [x for x in df_pre.columns if 'mem' in x]


# Next we want to make a list with any other columns that might be useful. This will consist of the manufacturer and bus columns. We want the manufacturer column as sometimes companies come out with patented technologies. It is possible that one manufacturer has exclusive rights to a certain type of RAM. Notice how bus is different from memBusWidth, this is because the bus column corresponds to the actual connections from the card to the motherboard, rather than the RAM to the card. I suspect as the bus gets newer (i.e. PCIe 3 or 4) the RAM should also get newer.

# In[4]:


usefulcols = ['manufacturer', 'bus']


# In[5]:


goodcols = memcols + usefulcols


# In[6]:


df = df_pre[goodcols].dropna()
df


# Taking a quick look at the shape of the original Dataframe against the shape of our new Dataframe shows that we did not lose that many rows, so we still have ample data to work with.

# In[7]:


df_pre.shape , df.shape


# ## Encoding Non-Numeric Columns

# The best way for us to predict the type of RAM is to use a decision tree, but before we can get to that we will need to do a little more preparation. Sikitlearn's decision tree algorithm requires numeric values as inputs and the bus and manufacturer columns are strings. Fortunately we can use Sikitlearn's OneHotEncoder to turn our bus and manufacturer columns into features that can be used in the decision tree.

# In[8]:


encoder = OneHotEncoder()


# In[9]:


encoder.fit(df[['manufacturer', 'bus']])


# In[10]:


arr = encoder.transform(df[['manufacturer', 'bus']]).toarray()


# In[11]:


enccols= encoder.get_feature_names_out()


# In[12]:


df[enccols] = arr
df


# In[13]:


df.shape


# Notice how our Dataframe went from 6 all the way to 41 columns. Now we have plenty of features to use in a decision tree.

# ## Creating the Decision Tree

# Decision Tree's are prone to overfitting, we want our predictions to be as accurate as possible so we will split our data into a traning and test set that contains 80% and 20% of the data respectivly. We can then use these to gauge accuracy.

# In[14]:


features = df[[x for x in df.columns if is_numeric_dtype(df[x][0]) == True]]
target = df['memType']


# In[15]:


X_train, X_test, y_train, y_test = train_test_split(features,target, train_size=0.8, random_state=0)


# Notice here that I set the max leaf nodes to 300. This seems like a lot. However as you can see 100 leaf nodes the tree seems to be underfitting. While at 400 leaf nodes the tree begins to start overfitting. 

# In[16]:


clf = DecisionTreeClassifier(max_leaf_nodes=300, random_state=0)
clf_under = DecisionTreeClassifier(max_leaf_nodes=100, random_state=0)
clf_over = DecisionTreeClassifier(max_leaf_nodes=400, random_state=0)


# In[17]:


clf.fit(X_train, y_train)
clf_under.fit(X_train, y_train)
clf_over.fit(X_train,y_train)


# In[18]:


clf.score(X_train, y_train) , clf_under.score(X_train, y_train), clf_over.score(X_train, y_train)


# In[19]:


clf.score(X_test,y_test) , clf_under.score(X_test,y_test), clf_over.score(X_test,y_test)


# As you can see from the first item in each tuple, on the training set we are getting around 94% Accuracy. While on the test set we are sitting at about 84% accuracy.

# ## Checking the Importances

# Now I am curious to see which features were the most important. I would assume that the features that have 'mem' in them are the most important. Although I hope the columns we encoded had a large impact.

# In[20]:


Importances = pd.Series(clf.feature_importances_, index=clf.feature_names_in_)


# In[21]:


df_imp = pd.DataFrame()
df_imp['Feature Name'] = clf.feature_names_in_
df_imp['Importance (%)'] = (100 * clf.feature_importances_).round(2)


# In[22]:


df_imp


# ## Graphing the Importances
# 
# Let's make a few charts to get a better look at the Importances. Out of curiosity I wanted to see how many times each item in the Bus and Manufacturer column appeared in the Dataframe. Let's find out if rarer features had more impact on the decision tree.
# 
# First lets make a quick Pie chart as that will be the easiest to read, also our importances should add up to 100% making it the perfect fit.

# In[23]:


select = alt.selection_single(fields=['Feature Name'] , bind='legend')


# In[24]:


c1 = alt.Chart(df_imp).mark_arc().encode(
    theta=alt.Theta(field='Importance (%)', type='quantitative'),
    color=alt.Color(field='Feature Name', scale=alt.Scale(scheme='category20')),
    tooltip=['Feature Name','Importance (%)'],
    opacity=alt.condition(select, alt.value(1), alt.value(0.2))
).add_selection(
    select
)


# Now let's work on finding out how many times each feature from the encoded columns appears in the Dataframe.
# 
# We can use some simple list comprehension to make a cool list of all the features from the encoded columns. Then some for loops to create two new lists, one of how many times each feature shows up in the Dataframe and the other to make our cool list look like the Feature Name colum of df_imp.

# In[25]:


coollist = [x for x in df['manufacturer'].unique()] + [x for x in df['bus'].unique()]


# In[26]:


howdy = []
for x in coollist:
    if x in [x for x in df['manufacturer'].unique()]:
        y = (df['manufacturer'] == x).sum()
    else:
        y = (df['bus'] == x).sum()
    howdy.append(y)


# In[27]:


coollist2=[]
for x in coollist:
    if x in [x for x in df['manufacturer'].unique()]:
        y = 'manufacturer_' + x
    else:
        y = 'bus_' + x
    coollist2.append(y)


# Now we can make a new Dataframe so we can create a Bar chart that will sync up with the Pie chart.

# In[28]:


df_bar = pd.DataFrame(list(zip(coollist2,howdy)), columns=['Feature Name','Occurance'])


# In[29]:


c2 = alt.Chart(df_bar).mark_bar().encode(
    x='Feature Name',
    y='Occurance',
    color='Feature Name',
    tooltip=['Feature Name','Occurance'],
    opacity=alt.condition(select, alt.value(1), alt.value(0.2))
).add_selection(
    select
)


# Looking at the Pie chart we can see that indeed the columns with 'mem' in them are the most important. It seems the columns we encoded were not all that important which is disappointing. It also seems my theory of how newer PCIe slots corresponding to newer RAM types was also wrong.
# Hovering over a slice or a bar with your mouse will show you the name of the slice or bar and importance/occurance. Clicking on a Feature Name in the legend will hilight it on both the Pie and Bar chart.

# In[30]:


alt.vconcat(c1,c2)


# After looking through some feature names, it is obvious that how often a feature appears in df has no corilation to the importance in the pie chart. Two key examples are the manufacturer_NVIDIA and the bus_PCIe 4.0 x16 features. They both share about the same importance 2.48 and 3.95 respectivly, although bus_PCIe 4.0 x16 appears far less than manufacturer_NVIDIA.

# ## Attempting to Visualize the Decision Tree
# Let's make a quick graph with memSize as the x-axis, memBusWidth as the y-axis, and memClock as the z axis. We are then going to need to zoom in quite a bit, so we can better see some data, rather than one big clump. You can actually explore this chart by using scroll wheel to zoom, right click to pan, and left click to rotate.
# 

# In[31]:


fig = px.scatter_3d(df, 
                    x = 'memSize', 
                    y = 'memBusWidth', 
                    z = 'memClock',
                    color = 'memType')


fig.update_layout(title='Full View')

camera = dict(eye=dict(x=2, y=1, z=0.5))


fig.update_layout(scene_camera=camera)
fig.show()


# Exploring the chart will show that colors seem to be clumped into rectangles. What a decision tree would attempt to do is draw planes in such a way that it isolates each of these colors. Note that our decision tree is a bit different as it has more than three features, However a three-dimensional chart is a lot easier to read than a 41 dimensional chart.
# 
# You can easily spot an isolated rectangle by rotating around the points in light blue. Let's restrict the size of the chart, so we can get a better look.

# In[32]:


fig = px.scatter_3d(df, 
                    x = 'memSize', 
                    y = 'memBusWidth', 
                    z = 'memClock',
                    color = 'memType')


fig.update_layout(
    title='Zoomed View',
    scene = dict(
        xaxis = dict(range=[3,17],),
        yaxis = dict(range=[1000,4100],),
        zaxis = dict(range=[675,1230],),),
    )


camera = dict(eye=dict(x=0.5, y=2, z=0.1))


fig.update_layout(scene_camera=camera)
fig.show()


# In this new chart all the points except for one correspond to the memType HBM2. So our decision tree should predict HBM2 if we give it features within the range of the chart. Let's quickly make up some features and see if it will guess correctly.

# In[33]:


data = np.zeros((100,len(clf.feature_names_in_)))
df_guess = pd.DataFrame(data , columns=clf.feature_names_in_)


# In[34]:


for i in range(0,100):
    df_guess.iloc[i,0] = rng.randint(3,7)
    df_guess.iloc[i,1] = rng.randint(100,4100)
    df_guess.iloc[i,2] = rng.randint(675,1230)
    x = rng.randint(3,10)
    y = rng.randint(11,37)
    df_guess.iloc[i,x] = 1
    df_guess.iloc[i,y] = 1


# In[35]:


df_guess


# Now we have 100 fake memory chips to work with, let's see how many turn out to have HBM2 as their memType.

# In[36]:


HBM2=[]
HBM2e=[]
other=[]

for i in range(0,100):
    tmp = clf.predict(df_guess.iloc[[i]])
    if tmp[0] == 'HBM2': 
        HBM2.append(1)
    elif tmp[0] == 'HBM2e':
        HBM2e.append(1)
    else:
        other.append(1)

print(f'''{len(HBM2)}% of the fake memory chips were of the type HBM2
{len(HBM2e)}% of the fake memory chips were of the tyoe HBM2e
{len(other)}% of the fake memory chips were neither HBM2 or HBM2e''')


# As you can see most of the fake chips did in fact be predicted to be HBM2. However, the ones that were not predicted as HBM2 were not likely to be predicted as HBM2e (the one green dot in the figure above).
# 
# After re-executing the code a couple of times, I have noticed that the percentage of fake memory chips that were predicted to be HBM2 seems to vary from around 50-60%, the percentage of HBM2e seems to vary from 0-5% and the percentage of others seems to vary from 40-50%

# ## Summary
# 

# It seems we were in fact able to use machine learning to guess the type of memory in a GPU using its specifications and some extra details about the GPU. As it turns out the clock speed of the memory is the biggest factor, and the memories buswidth and size combined are just as important. We also found out that the manufacturer and the bus do have an impact on the type of memory albeit to a lesser extent than the specs of the memory.

# ## Resources

# **GPU Dataset:**
# https://www.kaggle.com/datasets/alanjo/graphics-card-full-specs
# 
# **Plotly 3d Charts:**
# https://www.geeksforgeeks.org/3d-scatter-plot-using-plotly-in-python/?ref=rp
# https://plotly.com/python/3d-axes/
# https://plotly.com/python/3d-camera-controls/
# 
# **Altair Built in Charts (with documentation):**
# https://altair-viz.github.io/gallery/index.html has both pie and interactive chart examples

# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=eed43daa-d12b-427d-a659-e309fe058440' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>
