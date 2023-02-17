#!/usr/bin/env python
# coding: utf-8

# # Week 10 Wednesday

# ## Announcements
# 
# * Please fill out a [course evaluation](https://evaluations.eee.uci.edu/assigned/action_list) if you haven't already!  (I think that link will work; let me know if it doesn't.)
# * I have office hours at 1pm today, in my office, RH 440J.
# * If you're behind on the video quizzes, try to catch up today.  I plan to convert them to "practice quizzes" and post the total video quiz score Thursday morning.
# * General plan for lectures this week: About 20 minutes lecture, then time to work on the course project.
# * I plan to have one Zoom office hour during our scheduled final exam time, 10:30am-11:30am on Monday, December 5th.

# ## Some random project advice
# 
# * Where to get ideas?
# 
# Browsing Kaggle is the most fun, but it might also be overwhelming.  **Browsing our class worksheets** would be an equally good option.  I'll be very happy if your project shows that you understood material from our worksheets.
# 
# * What is realistic?
# 
# A rule of thumb is that, if a human expert could not do something, then you shouldn't expect a Machine Learning algorithm to be able to do it.  For example, a human expert can predict the price of a house quite accurately.  A human expert probably cannot predict the zip code of a house.
# 
# That rule of thumb is for advanced machine learning models.  You shouldn't expect something written in a short period of time to match a human expert.  Maybe more realistic is to try to do better than random guessing or some other simple baseline algorithm (like always predicting the median value).
# 
# * What if my project is too short?
# 
# One option is just load a different dataset and do something else.  (Don't do the same thing twice... that's not a good use of time.)
# 
# * What should I do to get a good grade?
# 
# Explain what you're doing clearly (in markdown cells, not Python comments) and show me what you learned in Math 10.  My favorite projects are the ones that clearly use the Math 10 material.
# 
# * Can you say more about references?
# 
# You don't have to reference my lecture notes, but basically everything else should be referenced (even if you make changes to it).  Provide a precise link when possible using this markdown syntax: `[text to display](http://www.uci.edu)` which will result in this: [text to display](http://www.uci.edu).  Ask on Ed Discussion if you're unsure about anything.

# ## More practice with the Spotify dataset
# 
# We also used this dataset on Monday.

# In[1]:


import pandas as pd
import altair as alt


# In[2]:


df = pd.read_csv("spotify_dataset.csv", na_values=" ").dropna(axis=0).copy()


# In[3]:


alt.Chart(df).mark_circle().encode(
    x="Energy",
    y="Danceability",
    color=alt.Color("Valence", scale=alt.Scale(scheme="spectral", reverse=True)),
    tooltip=["Artist", "Song Name"],
)


# * Load the data from [this page](https://gist.github.com/mbejda/9912f7a366c62c1f296c) on GitHub and name the result `df2`.
# 
# There are lots of ways to get this data (for example, you could probably copy and paste it into Excel, and then save the Excel file as a csv file).  We'll see a surprisingly easy way.

# If you follow the above link, you will notice a button that says `raw` near the table.  If you click that button, you will get the contents of the csv file, without any formatting.  Also notice that the resulting url ends in csv.  We save that url as a string here.

# In[4]:


# URL for the raw data
url = "https://gist.githubusercontent.com/mbejda/9912f7a366c62c1f296c/raw/dd94a25492b3062f4ca0dc2bb2cdf23fec0896ea/10000-MTV-Music-Artists-page-1.csv"


# We can now load the data directly from that website.  Notice how we do not need to download the csv file to our computer first.

# In[5]:


df2 = pd.read_csv(url)


# In[6]:


df2.head()


# * Save that dataset as a csv file, in case it later disappears from GitHub.
# 
# When using this `to_csv` method, I almost always use `index=False`, because most of the datasets I work with do not contain any interesting information in the index.

# In[7]:


df2.to_csv("from_github.csv", index=False)


# * Merge the result into the Spotify dataset.
# 
# Remember this `merge` method in case you find yourself wanting to combine multiple datasets.

# In[8]:


df.sample(3)


# In[9]:


df2.sample(3)


# Notice how both DataFrames contain the artist name, under the columns named "Artist" and "name", respectively.  We try to merge these together.  There is not an error, but the resulting DataFrame is empty.  (The `how="inner"` tells pandas to only keep the values that appear in both DataFrames.)

# In[10]:


df.merge(df2, left_on="Artist", right_on="name", how="inner")


# Let's look more closely at `df2`.  The first "name" that appears is Adele.  Does Adele appear in the other DataFrame, `df`?

# In[11]:


df2.head()


# The following is counter-intuitive to me.  If you ask if something is in a pandas Series, pandas will check if it occurs in the *index* of that Series.

# In[12]:


"Adele" in df["Artist"]


# Here is a more explicit way to check the same thing.

# In[13]:


"Adele" in df["Artist"].index


# What we really want is to check if Adele occurs in the values of the pandas Series.  (Notice how we do not put parentheses after `values`.  This is different from what you would do with a pandas dictionary.)

# In[14]:


"Adele" in df["Artist"].values


# We're now back where we started.  Adele seems to occur in both DataFrames.  Why didn't our `merge` work?
# 
# **Added after lecture**.  I see I made a mistake here.  I should have done `df2["name"].values`, like we were just discussing above!  I didn't notice the mistake because I was expecting to get `False`.

# In[15]:


"Adele" in df2["name"]


# Let's look more closely at the top-left entry.  Notice how there are spaces on either side.

# In[16]:


df2.loc[0, "name"]


# There is a Python string method, `strip`, that, if you don't pass any arguments, will remove whitespace from either end of a string.

# In[17]:


" chris    ".strip()


# We want to apply that method to each entry.  Using `map` is a good idea, but the following does not work, because there is no Python function `strip`.

# In[18]:


df2["name"].map(strip)


# I expected the following to work, but it didn't because of missing values.

# In[21]:


df2["name"].map(lambda x: x.strip())


# Let's remove the rows where the "name" value is missing.  (I didn't want to use `dropna`, because I only care about the "name" column.  If something is missing in a different column, I don't want to remove that row.)

# In[22]:


df3 = df2[~df2["name"].isna()]


# Now we can use the `map` method.

# In[23]:


df3["name"].map(lambda x: x.strip())


# Let's make a new DataFrame.

# In[24]:


df4 = df3.copy()


# Let's replace the "name" column with the stripped version.

# In[25]:


df4["name"] = df3["name"].map(lambda x: x.strip())


# Now we can finally perform our merge.  If you scroll all the way to the right, you will see that the web links have been added to the far right side.
# 
# It was some work to get to this stage, and that's what a lot of data science is like.  There are sayings to the effect of, "a data scientist spends 80% of their time cleaning the data".  In this case, the cleaning we did was removing the rows where the "name" value was missing, and also removing the white space from the name strings.

# In[27]:


df5 = df.merge(df4, left_on="Artist", right_on="name", how="inner")
df5


# Here we add a new encoding channel, the `href` channel.  If you click on one of the points, Altair will open the link that is in the "twitter" column for that row.

# In[29]:


# On Deepnote, alt+click or command+click to open links
alt.Chart(df5).mark_circle().encode(
    x="Energy",
    y="Danceability",
    color=alt.Color("Valence", scale=alt.Scale(scheme="spectral", reverse=True)),
    tooltip=["Artist", "Song Name"],
    href="twitter"
)

