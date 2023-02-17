#!/usr/bin/env python
# coding: utf-8

# # Week 4 Monday

# ## Announcements
# 
# * Worksheets 5 and 6 due tomorrow before discussion section.
# * Videos and video quizzes posted; due Friday before lecture.
# * Midterm next Wednesday, October 26th, during lecture.  Similar question style to the in-class quizzes, but some questions may be longer.  Note cards available Wednesday (Oct 19th).  Sample midterm posted by the end of this week.

# ## Introduction
# 
# Say we want to investigate how the bill length of penguins differs between male penguins and female penguins.  How can we investigate this using a *facet chart* in Altair? 

# In[1]:


import pandas as pd
import altair as alt
import seaborn as sns


# We are going to use a shortcut, `dropna`, to drop rows with missing values.  See this week's videos for an explanation of why `dropna(axis=0)` will drop the rows (as opposed to columns).  In Math 10, we will basically always drop rows instead of dropping columns.  That is because each row typically represents a data point.
# 
# If instead we were to drop columns with missing values, we would wind up with a very weird dataset.  This is what it would look like if we dropped the columns with missing values from this dataset.

# In[2]:


# Weird
# Removing columns with missing values.  See this week's videos for why `axis=0` is correct
df = sns.load_dataset("penguins").dropna(axis=1)


# Only two remaining columns.

# In[3]:


df


# Here is the usual usage.  This looks much more typical.

# In[4]:


# Removing rows with missing values.  See this week's videos for why `axis=0` is correct
df = sns.load_dataset("penguins").dropna(axis=0)


# In[5]:


df


# Our charts below will be variants of this one.  We eventually want to see whether male penguins or female penguins have longer bill length.  We cannot answer that from this image, because the sex of the penguins is not shown in this chart.

# In[6]:


alt.Chart(df).mark_circle().encode(
    x=alt.X("bill_depth_mm", scale=alt.Scale(zero=False)),
    y=alt.Y("bill_length_mm", scale=alt.Scale(zero=False)),
    color="species",
).properties(
    height=500,
    width=500,
    title="Penguins"
)


# ## Facet charts

# * Investigate bill length using a scatter plot together with `facet`.

# Here are the columns we have to work with.  We are most interested in the "bill_length_mm" column and the "sex" column.

# In[8]:


df.columns


# There was a question about what the `height` and `width` keyword arguments are doing.  Here we change the `height` from `200` to `20`; notice how the chart gets squashed.

# In[7]:


alt.Chart(df).mark_circle().encode(
    x=alt.X("bill_depth_mm", scale=alt.Scale(zero=False)),
    y=alt.Y("bill_length_mm", scale=alt.Scale(zero=False)),
    color="species",
).properties(
    height=20,
    width=200
).facet(
    column="sex"
)


# Back to the original height.  This facet chart already gives us a good understanding of how the bill length differs between male and female penguins across various species.

# In[8]:


alt.Chart(df).mark_circle().encode(
    x=alt.X("bill_depth_mm", scale=alt.Scale(zero=False)),
    y=alt.Y("bill_length_mm", scale=alt.Scale(zero=False)),
    color="species",
).properties(
    height=200,
    width=200
).facet(
    column="sex"
)


# If put the male and female charts in different rows instead of different columns, it is easier to compare the bill depths but it becomes harder to compare the bill lengths.  Today we are primarily interested in bill length.

# In[9]:


alt.Chart(df).mark_circle().encode(
    x=alt.X("bill_depth_mm", scale=alt.Scale(zero=False)),
    y=alt.Y("bill_length_mm", scale=alt.Scale(zero=False)),
    color="species",
).properties(
    height=200,
    width=200
).facet(
    row="sex"
)


# * Investigate bill length using a bar chart together with `facet`.

# Here we use the exact same code as above, but switching to a bar chart with `mark_bar` instead of `mark_circle`.

# In[10]:


alt.Chart(df).mark_bar().encode(
    x=alt.X("species", scale=alt.Scale(zero=False)),
    y=alt.Y("bill_length_mm", scale=alt.Scale(zero=False)),
    color="species",
).properties(
    height=200,
    width=200
).facet(
    column="sex"
)


# I don't think our size specifications (`height` and `width`) are improving the appearance of these bar charts; I think the default values look better.  So let's remove that portion of the chart.

# In[11]:


alt.Chart(df).mark_bar().encode(
    x=alt.X("species", scale=alt.Scale(zero=False)),
    y=alt.Y("bill_length_mm", scale=alt.Scale(zero=False)),
    color="species",
).facet(
    column="sex"
)


# Let's also bring the bars back to their default setting, of including zero for quantitative data types (like the y-axis in this example).

# In[12]:


alt.Chart(df).mark_bar().encode(
    x=alt.X("species"),
    y=alt.Y("bill_length_mm"),
    color="species",
).facet(
    column="sex"
)


# Notice how the above charts seem to suggest that female chinstrap penguins have a longer bill than male chinstrap penguins.  That is deceptive though, because these charts have one bar for each penguin, layered (not stacked) on top of each other.  So in fact, all this means is that one female chinstrap penguin has a longer bill.  Go back up to the scatter plots and see if you believe that one female penguin has a longer bill.
# 
# If we instead specify that we just want to know the `mean` of the bill length, then we get a more representative image.  The following is the first time we can tell from the bar chart that the average bill length is higher for male penguins than for female penguins.

# In[13]:


alt.Chart(df).mark_bar().encode(
    x=alt.X("species"),
    y=alt.Y("mean(bill_length_mm)"),
    color="species",
).facet(
    column="sex"
)


# I actually think it's more useful to use some different encodings in this example.  Let's start by making the facet charts by "species" instead of by "sex".  The following chart does not include the sex information (and overall it doesn't look very good).

# In[16]:


alt.Chart(df).mark_bar().encode(
    x=alt.X("species"),
    y=alt.Y("mean(bill_length_mm)"),
    color="species",
).facet(
    column="species"
)


# Here we specify that the x-axis should use the "sex" column.  I think this is the most readable of the charts, in terms of ease of comparing average bill length between different species of penguins.

# In[17]:


alt.Chart(df).mark_bar().encode(
    x=alt.X("sex"),
    y=alt.Y("mean(bill_length_mm)"),
    color="species",
).facet(
    column="species"
)


# Because we've gotten rid of the keyword arguments (like `scale = alt.Scale(...)`), we can get rid of the `alt.X` and `alt.Y` portions (we need to keep the column names).

# In[18]:


alt.Chart(df).mark_bar().encode(
    x="sex",
    y="mean(bill_length_mm)",
    color="species",
).facet(
    column="species"
)


# ## Facet charts "by hand" using `groupby`
# 
# * Make a similar chart to the above bar chart using `groupby` and `hconcat`.

# Here is a first attempt at making three bar charts (one for each species) and putting those charts into a list named `chart_list`.  (Python is very flexible about what can go into a list or a tuple.  As far as I know, any Python object can go into a list or a tuple.)

# In[19]:


chart_list = []

for species, df_sub in df.groupby("species"):
    c = alt.Chart(df).mark_bar().encode(
        x="sex",
        y="mean(bill_length_mm)",
    )
    chart_list.append(c)


# Let's try to display some of these charts horizontally, using `alt.hconcat` (the "h" stands for "horizontal".)  Because there are 3 species of penguins, we repeat the code inside the for loop 3 times, so there are only 3 charts in `chart_list`, so the following code raises an error.  (For testing, it is very helpful to initialize `chart_list = []` in the same cell as the for loop, so that it gets reset every time we run this cell.)

# In[20]:


alt.hconcat(chart_list[0], chart_list[1], chart_list[2], chart_list[3])


# The following works, but the charts all look the same.  That is because we were using the original DataFrame `df`, so each of the charts `c` was exactly the same.

# In[21]:


alt.hconcat(chart_list[0], chart_list[1], chart_list[2])


# Here we use `df_sub` instead of `df`, and we also add a title.  
# 
# (See the notes from Friday of last week if you're confused about what `species` and `df_sub` represent in the line `for species, df_sub in df.groupby("species"):`.  On Friday we talked about what happens when we iterate over a pandas GroupBy object.  On Friday we were using the cars dataset.)

# In[22]:


chart_list = []

for species, df_sub in df.groupby("species"):
    c = alt.Chart(df_sub).mark_bar().encode(
        x="sex",
        y="mean(bill_length_mm)",
        color="species"
    ).properties(
        title=species
    )
    chart_list.append(c)


# In[23]:


alt.hconcat(chart_list[0], chart_list[1], chart_list[2])


# It is definitely cumbersome to write out `chart_list[0], chart_list[1], chart_list[2]`.  It would be nice if we could just pass the list as our argument, but `alt.hconcat` does not accept a list as an argument; it wants charts as its arguments.

# In[37]:


alt.hconcat(chart_list)


# Luckily there is a Python abbreviation (not an Altair abbreviation) for "unpacking" the elements in the list and using them as inputs.  All we have to do is put a `*` before the name of the list.

# In[24]:


# same as alt.hconcat(chart_list[0], chart_list[1], ...)
alt.hconcat(*chart_list)

