#!/usr/bin/env python
# coding: utf-8

# # Week 10 Monday

# ## Announcements
# 
# * I have office hours at 11am today, next door in ALP 3610.
# * Videos and video quizzes due.
# * Worksheets 15 and 16 due Tuesday by 11:59pm.
# * There won't be a "Quiz 7".
# * General plan for lectures this week: About 20 minutes lecture, then time to work on the course project.

# ## Don't expect conclusive results in the course project
# 
# If you're coming up with your own research question, as opposed to following a tutorial, you shouldn't expect to get conclusive (maybe not even interesting) results... that's just how research goes a lot of the time.
# 
# Today I want to use the Spotify dataset.  I think that's a great dataset for Exploratory Data Analysis (EDA).  I've tried for many hours to find an interesting Machine Learning question we can answer using this dataset (like classification: "Predict if the artist is Taylor Swift or Billie Eilish" or regression: "Predict the number of streams of a song"), but have never gotten any convincing results.
# 
# For the course project, you can decide for yourself if you'd rather investigate your own question or if you'd rather investigate someone else's question and be more confident that you will get conclusive results.  Both are good options.

# ## Preparing the Spotify dataset for K-means clustering
# 
# General guiding question: If we perform K-means clustering on the Spotify dataset, do songs by the same artist tend to appear in the same cluster?
# 
# The first step is to choose what columns we will use.  They need to be numeric columns and they need to not have any missing values.

# In[2]:


import altair as alt
import pandas as pd
from pandas.api.types import is_numeric_dtype


# Remember that missing values in this dataset are represented by a blank space `" "` (not an empty string `""` which would be fine).  Here we also drop the rows containing missing values.  Sometimes that is too drastic (like on the Titanic dataset), but here it only removes about 10 rows.

# In[3]:


df = pd.read_csv("spotify_dataset.csv", na_values=" ").dropna(axis=0).copy()


# Let's choose which columns (aka features, aka input variables) we want to use for our clustering.  They should all be numeric.  (If you want to use a categorical feature, first use `OneHotEncoder`.)

# In[5]:


# All numeric columns
num_cols = [c for c in df.columns if is_numeric_dtype(df[c])]
num_cols


# Let's use the columns from "Popularity" to the end.  (I don't want to use anything related too closely to the artist.)
# 
# I don't think we've used the `index` method before on a list.  Use the list method `index` to find at what index "Popularity" occurs.

# In[7]:


i = num_cols.index("Popularity")
i


# Make a list `cols` which contains the numeric column names from "Popularity" to the end.

# In[8]:


cols = num_cols[i:]
cols


# Here is a reminder of what the data in this dataset looks like.  Notice in particular how big the numbers are in the "Duration (ms)" column.  It would be a bad idea to use clustering on this dataset without scaling.

# In[9]:


df.head(3)


# ## K-means clustering review
# 
# We saw K-means clustering at the very beginning of the Machine Learning portion of Math 10.  (It is the only example of unsupervised learning we have discussed.)  Let's review K-means clustering.  Think about how it compares and contrasts to the supervised Machine Learning we have done (linear and polynomial regression, decision trees for classification and regression, random forests).
# 
# Remember that scaling is typically very important for K-means clustering.  If two of your columns have the same unit (like money spent on rent and money spent on food), then maybe you don't need to rescale those, but if the units for the columns are different (like money spent on rent and distance traveled to work), then I think it's essential to rescale.  (Scaling does not have any impact on decision trees, I don't think.  Scaling before linear regression can be useful or not useful depending on the context.  If you want to know which feature is the most important, I think it makes sense to rescale.  If you want to be able to interpret the precise coefficients that show up, then I don't think it makes sense to rescale.)

# In[10]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# Recommended exercise: try to do what we're doing below without using `Pipeline`.  I think you'll find that it involves much more typing.

# In[11]:


pipe = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(n_clusters=6))
    ]
)


# Remember that we can't use all the columns, because K-means clustering requires numerical values in all the columns.

# In[12]:


pipe.fit(df)


# Here we use a portion of the numeric columns.  (Eventually I want to see how songs by a single artist get divided into different clusters, so I want to remove columns like especially "Artist Followers" which will be extremely connected to the artist.)

# In[13]:


pipe.fit(df[cols])


# Remember that K-means is an algorithm from unsupervised learning.  That corresponds in our `fit` call that we only gave an input `X`, not any true labels `y`.

# ## Most frequent artists
# 
# Which 25 artists appear most often in the dataset?  Use `value_counts`, and take advantage of the fact that by default the results are sorted.

# In[14]:


df["Artist"].value_counts()


# I think we used this trick once before.  We get a pandas Index (similar to a list) containing the top 25 artists.

# In[15]:


top_artists = df["Artist"].value_counts().index[:25]
top_artists


# ## Making an Altair chart
# 
# I don't know if we'll have time, but let's try to make a chart like [this one from the Altair examples gallery](https://altair-viz.github.io/gallery/normalized_stacked_bar_chart.html) so we can see which artists' songs fall into which clusters.  Only use the 25 most frequently occurring artists.

# Here are the last 5 rows in the DataFrame.

# In[17]:


df[-5:]


# We will use the following Boolean Series (a pandas Series with `True` and `False` for its values) to perform Boolean indexing.  Notice for example how there is a `True` in the rows with labels 1551 and 1555.  Those artists (as we can see from the above slice) are Dua Lipa and Taylor Swift.

# In[18]:


bool_ser = df["Artist"].isin(top_artists)
bool_ser


# Now we get the rows whose corresponding artist is among the top 25.  We also use `copy` so we can add a column to it later without any warnings showing up.

# In[20]:


df_top = df[bool_ser].copy()


# In[22]:


df_top["pred"] = pipe.predict(df[cols])


# We now add a new column to `df_top` that contains the cluster numbers.  Notice how we fit the clustering algorithm using every row in `df`, but we are only using `predict` on 504 of the rows.  It's essential to use the same columns, but there's no need to use the same rows.

# In[26]:


df_top["pred"] = pipe.predict(df_top[cols])


# There is now a new column on the right side, containing the cluster values.

# In[24]:


df_top


# Here is the adaptation of the Altair code from the above link.  The `stack="normalize"` portion makes sure the total width of the chart is constant.
# 
# I don't know enough about these artists to see any clear interpretation of the results.  Notice for example how the songs by Polo G and DaBaby are almost all in the same cluster (cluster 0), whereas no songs by Kid Cudi and Lady Gaga are in that cluster.

# In[25]:


alt.Chart(df_top).mark_bar().encode(
    x=alt.X('count()', stack="normalize"),
    y='Artist',
    color='pred:N'
)


# ## Aside: Idea for extra component for your course project

# Random idea for an extra component for the project: use geojson data in Altair to make maps: [randomly chosen reference](https://github.com/altair-viz/altair/issues/1612#issuecomment-511830524).  Here is a more systematic reference: [UW Visualization Curriculum](https://uwdata.github.io/visualization-curriculum/altair_cartographic.html).  As far as I know, there aren't yet any geojson examples in the Altair documentation (there should be within the next few months).
