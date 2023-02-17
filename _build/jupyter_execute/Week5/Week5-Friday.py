#!/usr/bin/env python
# coding: utf-8

# # Week 5 Friday

# ## Announcements
# 
# * Videos for next week posted soon.
# * Next worksheets will be given out Monday/Wednesday.  Due Tuesday of Week 7.
# * No in-class quiz during Week 6.
# * Midterms will probably be returned early Week 6.
# * (Totally optional of course.)  See attached flyers for two recruitment flyers I was asked to share.

# ## Introduction to Machine Learning
# 
# Slides available in the <a href="../Week5-ML.pdf">course notes</a>.

# ## K-means clustering using scikit-learn
# 
# Using `KMeans` from scikit-learn, cluster the penguins data using the columns "bill_length_mm" and "flipper_length_mm".

# In[2]:


import pandas as pd
import altair as alt
import seaborn as sns


# In[3]:


df = sns.load_dataset("penguins")


# In[4]:


col0 = "bill_length_mm"
col1 = "flipper_length_mm"


# There is a subtle mistake in the following.  We are using `"col0"` instead of `col0`, so Altair is looking for a column whose name is the string `"col0"`.

# In[5]:


alt.Chart(df).mark_circle().encode(
    x=alt.X("col0", scale=alt.Scale(zero=False)),
    y=alt.Y("col1", scale=alt.Scale(zero=False))
)


# Here is the correct chart.

# In[6]:


alt.Chart(df).mark_circle().encode(
    x=alt.X(col0, scale=alt.Scale(zero=False)),
    y=alt.Y(col1, scale=alt.Scale(zero=False))
)


# Let's try to divide that data into two clusters.  There is a general routine used with scikit-learn for Machine Learning.  The more comfortable you get with this routine, the more familiar you will be with some of the conventions of Object Oriented Programming.
# 1.  Import (this step only needs to be done once per session)
# 2.  Instantiate (i.e., create an object/instance of the class you just imported)
# 3.  Fit
# 4.  Predict (or transform).

# In[7]:


# import
from sklearn.cluster import KMeans


# If you don't specify the number of clusters, scikit-learn will use 8 clusters as its default value.  (It would be nice if scikit-learn would determine the number of clusters automatically based on the data, but it doesn't.)  Be sure to use different capitalizations for the object `kmeans` and the class we imported `KMeans`.  The convention is to use lower-case letters for our objects.

# In[8]:


# instantiate
kmeans = KMeans(n_clusters=2)


# In[9]:


# Example of Object Oriented Programming, use a special KMeans object
type(kmeans)


# Python error messages are not always clear, but this one is pretty clear (scroll to the bottom).

# In[10]:


kmeans.fit(df[[col0,col1]])


# We drop the missing values.  (Stop and convince yourself that `axis=0` is correct if we want to drop rows.  We are changing the rows axis.)  The `copy` is to prevent a warning later.

# In[11]:


df2 = df.dropna(axis=0).copy()


# The displayed output is new to me (at least how it is displayed on Deepnote).  Think of the following cell as changing the object `kmeans`.

# In[13]:


# Step 3: fit
kmeans.fit(df2[[col0,col1]])


# Your output may look different, because there is some randomness to the K-Means algorithm.  Especially the specific numbers could be swapped.

# In[14]:


# Step 4: predict
arr = kmeans.predict(df2[[col0, col1]])
arr


# The variable `arr` is a NumPy array.  It has length `333`.

# In[15]:


len(arr)


# That's not the same length as `df`, because we removed some rows from `df`.

# In[16]:


len(df)


# Here it's the same.

# In[17]:


len(df2)


# Let's try the routine, this time looking for 4 clusters.

# In[18]:


kmeans2 = KMeans(n_clusters=4)


# In[19]:


kmeans2.fit(df2[[col0,col1]])


# Notice how now we see 4 different values in the NumPy array, corresponding to the 4 clusters.

# In[20]:


arr2 = kmeans2.predict(df2[[col0,col1]])
arr2


# Let's make a new column in `df2` holding these cluster values.  (This next line would raise a warning if we hadn't used `copy` when we dropped rows with missing values.)

# In[21]:


df2["cluster"] = arr2


# The default encoding type for the "cluster" column is Quantitative (`:Q`), but that is probably the worst choice from among Quantitative, Ordinal, and Nominal.

# In[22]:


alt.Chart(df2).mark_circle().encode(
    x=col0,
    y=col1,
    color="cluster"
)


# The ordering of the cluster numbers 0,1,2,3 is not significant, so Nominal is by far the best choice of encoding type.
# 
# Notice how the clusters seem to be lying in horizontal bands.  Look at the scales of the x-axis and the y-axis.  The data along the x-axis covers about a range of 20, whereas the y-axis data covers a range of about 50.  This difference in magnitude is causing the clusters to lie along horizontal bands.  We will see how to get more appropriate clusterings next week.

# In[23]:


alt.Chart(df2).mark_circle().encode(
    x=col0,
    y=col1,
    color="cluster:N"
)


# Here is a zoomed in version where we remove the default inclusion of zero in the axes.

# In[26]:


alt.Chart(df2).mark_circle().encode(
    x=alt.X(col0, scale=alt.Scale(zero=False)),
    y=alt.Y(col1, scale=alt.Scale(zero=False)),
    color="cluster:N"
)

