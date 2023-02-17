#!/usr/bin/env python
# coding: utf-8

# # Week 6 Monday

# ## Announcements
# 
# * I have office hours after class at 11am, next door in ALP 3610.
# * Videos and video quizzes posted; due Friday before lecture.
# * Midterms should be available on Gradescope Tuesday.
# * Plan for today: Discussion of the K-means algorithm, followed by time to work on Worksheet 9.

# ## The K-means algorithm
# 
# Here we describe the K-means algorithm.  On the worksheet, you will use a `KMeans` object from scikit-learn, which will run this algorithm automatically.
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


# ## Making an artificial dataset for clustering

# In[2]:


from sklearn.datasets import make_blobs


# The `make_blobs` function returns two things, but we will only need the first one, which we assign to the variable `X`.  A convention in Python is to use underscore `_` for a variable name that won't be used.
# 
# For the keyword arguments to `make_blobs`:
# * Think of `n_samples` as the number of points (also the number of rows in the NumPy array).
# * Think of `n_features` as the number of dimensions (also the number of columns).
# * The argument `centers` refers to how many blobs will be made.
# * The `random_state=1` is to have reproducible random numbers.  (If you run this code, using the same parameters, especially the `random_state` parameter, you should get the exact same numbers in `X`.)

# In[5]:


# Notice centers does not equal n_clusters
X, _ = make_blobs(n_samples=500, n_features=2, centers=4, random_state=1)


# This `X` is a NumPy array.

# In[6]:


type(X)


# It has 500 rows and 2 columns.  Today we want to plot the points, so using 2 columns makes the most sense.  In the worksheet today, we will be working with higher-dimensional data.

# In[7]:


X.shape


# NumPy arrays share many similarities with pandas DataFrames, but the rows and columns are not labeled in a NumPy array.  For example, there is no attribute `X.columns`.

# In[8]:


X.columns


# Here are the first 5 rows in `X`.

# In[9]:


X[:5]


# ## The first iterations by hand
# 
# Let's use the following three points for our initial centroids.
# ```
# [[-4, -10], [-4, -8], [-10, -10]]
# ```

# Because we called the `make_blobs` function ourselves using `centers=4`, we secretly know there are 4 clusters.  But in real-world scenarios, we probably won't know how many clusters there are.  To emphasize that, we will try to divide the data into 3 clusters.

# In[10]:


n_clusters=3


# The first step of the K-means algorithm is to choose `n_clusters` random centroids (they don't actually have to be centers of anything).  Here are our guesses; any numbers would make the same amount of sense.

# In[11]:


centroids = np.array([[-4, -10], [-4, -8], [-10, -10]])


# In[12]:


centroids.shape


# So that we can plot using Altair, we will put our random data from `make_blobs` into a DataFrame.

# In[13]:


df_data = pd.DataFrame(X, columns=["x", "y"])


# We do the same thing for our centroids data.  This is a much smaller DataFrame (just 3 rows because we are looking for 3 clusters).

# In[14]:


df_centroids = pd.DataFrame(centroids, columns=["x","y"])


# Notice how the data does seem to lie in 4 clusters, as expected.

# In[15]:


c1 = alt.Chart(df_data).mark_circle().encode(
    x="x",
    y="y"
)

c1


# Here are our initial guesses for the centers.  We add some customization to make the points more visible.  (Notice how we are using `mark_point` instead of the usual `mark_circle`.

# In[17]:


c2 = alt.Chart(df_centroids).mark_point(
    size=500,
    shape='cross',
    filled=True,
    stroke="black",
).encode(
    x="x",
    y="y",
)

c2


# Here we see the two charts side-by-side.

# In[18]:


alt.hconcat(c1,c2)


# More helpful is to have the charts "layered" one on top of the other.

# In[19]:


alt.layer(c1,c2)


# We haven't really started the K-means clustering algorithm yet.  All we've done is choose our initial guesses for the centroids.  The next step is to assign each point to its nearest centroid.  Let's try to do that at first with a single point, the 20-th row of `X`.  We can access that 20-th row using `X[20]`.  (This would not work if `X` were a DataFrame, because then pandas would be looking for a column with label the integer `20`.)

# In[21]:


z = X[20]
z


# As a reminder, these are our centroids.  Notice how our point `z` is closest to the last of these three points, `centroids[2]`.

# In[22]:


centroids


# It's not obvious that subtracting a length 2 one-dimensional NumPy array from a 3-by-2 two-dimensional NumPy array should make sense, but it does, because of NumPy's rules of broadcasting.  (Those rules are tricky to wrap your head around at first; for now, just notice that the following works.)  For example, the `0.47...` in the lower-left entry corresponds to `-10 - -10.47...`.

# In[23]:


# trick uses what's called "broadcasting" in NumPy
centroids - z


# We now square each of these terms individually.  (Notice how squaring a NumPy array automatically performs this squaring operation elementwise.)

# In[24]:


(centroids - z)**2


# There's nothing special about the `**` notation, it's just the (strange) "usual" way to raise to a power in Python.

# In[25]:


6**2


# We next add up the squared differences along each row.  Take a minute and convince yourself that `axis=1` is the correct keyword argument to use in this case.

# In[26]:


((centroids - z)**2).sum(axis=1)


# If we really wanted to know the distance between `z` and each point in `centroids`, we should now take a square root of each element.  All we care about is, which is the smallest, so there is no need to take that square root.  We save some computational power by not taking the square root.
# 
# When we call `argmin`, we get as output `2`, for this particular choice of `z`.  This corresponds to what we said above, that `z` is closest to `centroids[2]`.  (I believe there is no `idxmin` in NumPy, because NumPy arrays don't have labels.)

# In[27]:


((centroids - z)**2).sum(axis=1).argmin()


# Let's turn that procedure we just did into a function.

# In[28]:


def closest_centroid_index(z, centroids):
    return ((centroids - z)**2).sum(axis=1).argmin()


# Here we check that the function is working as expected.

# In[29]:


closest_centroid_index(z, centroids)


# That function was applied to a single row, the row `X[20]`.  We want to apply it to every row.  (Take a minute to convince yourself that `axis=1` is the keyword argument to use.)  We can do that using `apply`.  It would be nice if we could just use
# ```
# df_data.apply(closest_centroid_index, axis=1)
# ```
# but because `closest_centroid_index` takes two input arguments, we won't be able to use this directly.
# 
# I had expected
# ```
# df_data.apply(lambda row: closest_centroid_index(row, centroids), axis=1)
# ```
# to work, but there is a subtle difference between `z`, which we used above, and a row, like we are using here.  I will have to think more about what is going wrong in the above case.  To get around it, we use `row.values`, which converts the pandas Series `row` into a NumPy array.  Since `z` above was a NumPy array, this is a natural choice.
# 
# The following values tell us, which of the three centroids is closest to each point (i.e., to each row).

# In[32]:


df_data.apply(lambda row: closest_centroid_index(row.values, centroids), axis=1)


# Here is a reminder of what `centroids` looked like.

# In[33]:


centroids


# And a reminder of what `df_data` looked like.

# In[34]:


df_data


# Let's store these values in a new column in `df_data`.

# In[33]:


df_data["first cluster"] = df_data.apply(lambda row: closest_centroid_index(row.values, centroids), axis=1)


# Here is the resulting chart, but it will be more meaningful below when we add in the centroids.

# In[34]:


c1 = alt.Chart(df_data).mark_circle().encode(
    x="x",
    y="y",
    color="first cluster:N"
)

c1


# Here is a shorthand for layering the charts together; instead of using `alt.layer(c1, c2)`, we use `c1+c2`.
# 
# For example, notice how there are just a few points, the blue points, which are closest to the lower-right centroid.

# In[35]:


c1+c2


# This already took most of class.  The next step would be to compute the centroids of these clusters, and then repeat the above procedure using the new centroids.  We continue repeating this two step process (compute centroids, assign points to the closest centroid) until the process terminates, meaning that none of the cluster values change.  We will see more steps of this process on Wednesday.
