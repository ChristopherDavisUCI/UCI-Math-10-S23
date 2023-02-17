#!/usr/bin/env python
# coding: utf-8

# # Week 6 Wednesday

# ## Announcements
# 
# * My usual office hours today are cancelled.  Replaced by today 11am next door in ALP 3610 (I think it will be available).
# * No curving on the midterm (Median 41.6, Mean 41.3) but if you score higher on Midterm 2 (Week 9), we will weight that midterm 35% and this midterm 15% (instead of 25% and 25%).
# * I forgot to post solutions to the midterm, will post them Friday!
# * Videos and video quizzes posted; due Friday before lecture.

# ## The K-means algorithm
# 
# Copied from Monday:
# 
# We choose the number of clusters we want to search for, `n_clusters`.  For this example, let's say `n_clusters=3` and let's say there are 500 points in the dataset.
# 
# 1.  Randomly choose 3 points, called *centroids*.
# 2.  For each of the 500 points in the dataset, assign it to the nearest centroid.  We have now divided the data into 3 clusters.
# 3.  Compute the centroids (also called averages, also called means) of each of these 3 clusters.  These 3 centroids are what is meant by the phrase *K-means* (in this case, `K = 3`).
# 4.  For each of the 500 points in the dataset, assign it to the nearest centroid.  Continue repeating this process (assign to a cluster, compute the centroid of each cluster) until the process terminates. (By terminates, we mean each point remains in the same cluster.)

# In[1]:


import numpy as np
import pandas as pd
import altair as alt

from sklearn.datasets import make_blobs


# ## Code from Monday
# 
# Here is the final code we used on Monday.  Notice that we use `centers=4` in `make_blobs`, which means there are 4 true centers, but we use 3 points in our `centroids` array (that is our guess for the number of centers).

# In[2]:


# Notice centers does not equal n_clusters
X, _ = make_blobs(n_samples=500, n_features=2, centers=4, random_state=1)


# In[3]:


centroids = np.array([[-4, -10], [-4, -8], [-10, -10]])
df_centroids = pd.DataFrame(centroids, columns=["x","y"])


# In[4]:


def closest_centroid_index(z, centroids):
    return ((centroids - z)**2).sum(axis=1).argmin()


# In[5]:


df_data = pd.DataFrame(X, columns=["x", "y"])


# In[6]:


df_data["cluster"] = df_data.apply(lambda row: closest_centroid_index(row.values, centroids), axis=1)


# In[7]:


c1 = alt.Chart(df_data).mark_circle().encode(
    x="x",
    y="y",
    color="cluster:N"
)

c2 = alt.Chart(df_centroids).mark_point(
    size=500,
    shape='cross',
    filled=True,
    stroke="black",
).encode(
    x="x",
    y="y",
)

c1+c2


# I want to give some more explanation of where the `row.values` in the `apply` part came from.  Here is a reminder of what our original random data looked like.  The NumPy array `X` was produced by `make_blobs`.

# In[12]:


X[:5]


# Here is a single point from among the 500 points.  It is a one-dimensional length-2 NumPy array.

# In[13]:


z = X[20] # NumPy version of df.iloc[20]
z


# Here is the version of that point from our DataFrame, `df_data`.

# In[14]:


df_data.iloc[20]


# Let's just look at the `"x"` and `"y"` portions of the pandas Series.

# In[15]:


df_data.iloc[20][["x","y"]]


# The NumPy array `z` has shape `(2,)`.  Notice how this tuple has length 1, which corresponds to `z` being a one-dimensional NumPy array.

# In[16]:


z.shape


# It was asked in class what would happen if we take the transpose of `z`.  (Aside: I've never understood why we use the attribute `.T` to take a transpose in NumPy, as opposed to using the method version, `.T()`.)  In this case, where our NumPy array is one-dimensional, there is no change when we take the transpose.

# In[17]:


z.T


# In[18]:


z


# In[19]:


z == z.T


# In[20]:


(z == z.T).all()


# ## Clustering with scikit-learn
# 
# Let's see how to make some of the same computations using scikit-learn.  With some more work above, we can get the exact same results by hand (but we didn't do that today).

# In[21]:


# import
from sklearn.cluster import KMeans


# Here is a reminder of our starting points.

# In[22]:


centroids


# Because we want to use 3 starting points, we need to use `n_clusters=3`.  The `max_iter=1` means we only want scikit-learn to perform 1 step of the K-means clustering algorithm.  (It will still go further than we went above.)  The `n_init=1` is more confusing.  Usually scikit-learn will repeat the entire clustering algorithm multiple times, and then choose the best one.  We are telling it to only perform the entire clustering algorithm once (and in fact, that is necessary because if we are always using the same starting points `centroids`, then we will always get the same results).

# In[23]:


# instantiate
kmeans = KMeans(n_clusters=3, init=centroids, max_iter=1, n_init=1)


# In[24]:


df_data[:5]


# In[25]:


df_data[["x","y"]]


# We want to fit the `KMeans` object `kmeans` using just the "x" and "y" columns.

# In[26]:


# fit
kmeans.fit(df_data[["x","y"]])


# Here are the predicted clusters.

# In[27]:


# predict
kmeans.predict(df_data[["x", "y"]])


# In[28]:


df_data["cluster-sklearn"] = kmeans.predict(df_data[["x", "y"]])


# In[29]:


df_data[:5]


# Let's now make the same chart as above, but this time coloring using the scikit-learn clusters.  The chart `c2` still shows the original starting points.

# In[30]:


c1 = alt.Chart(df_data).mark_circle().encode(
    x="x",
    y="y",
    color="cluster-sklearn:N"
)

c2 = alt.Chart(df_centroids).mark_point(
    size=500,
    shape='cross',
    filled=True,
    stroke="black",
).encode(
    x="x",
    y="y",
)

c1+c2


# Let's update the chart to show the next centers.  (Remember, each step of the K-means clustering algorithm, new centers are computed.)

# In[31]:


centroids2 = kmeans.cluster_centers_


# In[32]:


centroids2


# Let's make a DataFrame with that data.

# In[33]:


df_centroids2 = pd.DataFrame(centroids2, columns=["x","y"])


# Let's see the new positions of these centroids.  If you scroll up and look at our clusters that we found by hand on Monday, and look at the centers of those clusters, they are exactly the points that are drawn below.

# In[34]:


c1 = alt.Chart(df_data).mark_circle().encode(
    x="x",
    y="y",
    color="cluster-sklearn:N"
)

c2 = alt.Chart(df_centroids2).mark_point(
    size=500,
    shape='cross',
    filled=True,
    stroke="black",
).encode(
    x="x",
    y="y",
)

c1+c2


# The next step in the clustering algorithm would be to move these centers to the centers of the current clusters (for example, the cross in the red cluster would definitely move up, because there are so many more red points above it than below it).  The next step after that would be to assign new clusters, based on what centroid is closest.  (I use the words *centroid*, *center*, *mean*, and *average* interchangeably.)
# 
# Worksheet 10 is a little longer than our usual worksheets, but it has a very nice payoff, where at the end of Worksheet 10, you will have a chart with a slider you can drag to illustrate the K-means clustering algorithm.
