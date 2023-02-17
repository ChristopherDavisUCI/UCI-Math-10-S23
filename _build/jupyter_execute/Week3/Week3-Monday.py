#!/usr/bin/env python
# coding: utf-8

# # Week 3 Monday

# ## Announcements
# 
# * Worksheets 3 and 4 due before discussion on Tuesday.  (I tried to make Worksheet 5 shorter.)
# * In-class quiz tomorrow during discussion section.  Based on the material up to and including Worksheet 4.  Ask William for any precise details.
# * I have office hours after class today at 11am, next door in ALP 3610.  Please come by with any questions!
# * Videos and video quizzes for Friday are posted.

# ## Plotting based on the Grammar of Graphics
# 
# If you've already seen one plotting library in Python, it was probably Matplotlib.  Matplotlib is the most flexible and most widely used Python plotting library.  In Math 10, our main interest is in using Python for Data Science, and for that, there are some specialty plotting libraries that will get us nice results much faster than Matplotlib.
# 
# Here we will introduce the plotting library we will use most often in Math 10, Altair.  In today's Worksheet, you will see two more plotting libraries, Seaborn and Plotly.
# 
# These three libraries are very similar to each other (not so similar to Matplotlib, although Seaborn is built on top of Matplotlib), and I believe all three are based on a notion called the Grammar of Graphics. (Here is the book [The Grammar of Graphics](https://link.springer.com/book/10.1007/0-387-28695-0), which is freely available to download from on campus or using VPN.  There is also a widely used, and I think older, plotting library for the R statistical software that uses the same conventions, ggplot2.)
# 
# Here is the basic setup for Altair, Seaborn, and Plotly:
# 
# * We have a pandas DataFrame, and each row in the DataFrame corresponds to one observation (or one data point).
# 
# * Each column in the DataFrame corresponds to a variable (also called a dimension, or a field).
# 
# * To produce the visualizations, we encode different columns from the DataFrame into visual properties of the chart (like the x-coordinate, or the color).
# 
# Altair tries to choose default values that produce high-quality visualizations; this greatly reduces the need for fine-tuning.  But there is also a huge amount of customization possible.  As one example, here are the named [color schemes](https://vega.github.io/vega/docs/schemes/) available in Altair.

# In[1]:


import pandas as pd


# We will get a dataset from Seaborn.  Other than that, we won't use Seaborn in today's lecture.  Seaborn does show up on Worksheet 5.

# In[2]:


# Seaborn
import seaborn as sns


# Here are the datasets included with Seaborn.  Most of these are small (a few hundred rows), so they should be considered as "toy" datasets for practice.

# In[3]:


sns.get_dataset_names()


# Here is the syntax for loading a dataset from Seaborn.  The result is a pandas DataFrame.

# In[4]:


df = sns.load_dataset("mpg")


# In[5]:


df


# Here are the data types in this cars dataset.

# In[7]:


df.dtypes


# Now we will start working with Altair.

# In[6]:


import altair as alt


# The syntax will take some getting used to.  (And we'll see below that it can get quite a bit more complicated than this, depending on how much control you want to have over the chart.)
# 
# Think of the following as proceeding in three steps.
# 
# 1. We tell Altair what pandas DataFrame we will use. `alt.Chart(df)` 
# 
# 2. We tell Altair that we want to use circles (I think of them as disks) as the mark type.  In other words, this will be a scatter plot (as opposed to for example a bar chart or a line chart).  `.mark_circle()` 
# 
# 3. We tell Altair which columns, in this case the "weight" column and "horsepower" column we want to encode in which visual properties of the chart.  Here they are encoded as the x and y positions, respectively.
# 
# ```
# .encode(
#     x="weight",
#     y="horsepower"
# )
# ```

# In[8]:


# syntax will seem weird at first
alt.Chart(df).mark_circle().encode(
    x="weight",
    y="horsepower"
)


# The strength of this style of plotting (which is shared by Seaborn and Plotly) is that you can use many more encodings than just the x-coordinate and y-coordinate.  Here we encode the "cylinders" column from the DataFrame in the size of the points.
# 
# Much more important than memorizing this syntax is to understand how the data in `df` is reflected in the following chart.  (Pick a row in the DataFrame and try to find the corresponding point in the chart.  Does its size look correct?)

# In[9]:


# syntax will seem weird at first
alt.Chart(df).mark_circle().encode(
    x="weight",
    y="horsepower",
    size="cylinders"
)


# The following tooltip list means that when we hover our mouse over a point, we will see the values for the properties listed in the tooltip.

# In[11]:


# syntax will seem weird at first
alt.Chart(df).mark_circle().encode(
    x="weight",
    y="horsepower",
    color="origin",
    tooltip=["name", "weight", "horsepower"]
)


# A common mistake is to spell a column name incorrectly.  When that happens, you will receive the following error message.  (You have to read the last part to get a clue that you input an incorrect column name.)
# 
# > ValueError: year encoding field is specified without a type; the type cannot be inferred because it does not match any column in the data.

# In[12]:


# syntax will seem weird at first
alt.Chart(df).mark_circle().encode(
    x="weight",
    y="horsepower",
    color="year",
    tooltip=["name", "weight", "horsepower"]
)


# It should have been "model_year" instead of "year".

# In[13]:


df.columns


# Here is the default coloring used when we encode the "model_year" column in the color channel.

# In[14]:


# syntax will seem weird at first
alt.Chart(df).mark_circle().encode(
    x="weight",
    y="horsepower",
    color="model_year",
    tooltip=["name", "weight", "horsepower"]
)


# Let's see how to choose our own color scheme.  As a first step, making no change to the produced chart, we replace `color="model_year"` with `color=alt.Color("model_year")`.  The advantage of using this longer syntax is that we can pass keyword arguments to the `alt.Color` constructor, which will be used to customize the appearance.

# In[15]:


# syntax will seem weird at first
alt.Chart(df).mark_circle().encode(
    x="weight",
    y="horsepower",
    color=alt.Color("model_year"),
    tooltip=["name", "weight", "horsepower"]
)


# The thing we are customizing is the color scale.  I know that it is a lot of writing, but the benefit is that there is a huge amount of customization possible.  See the following for the possible named color schemes: [color scheme choices](https://vega.github.io/vega/docs/schemes/).

# In[16]:


# syntax will seem weird at first
alt.Chart(df).mark_circle().encode(
    x="weight",
    y="horsepower",
    color=alt.Color("model_year", scale=alt.Scale(scheme="goldgreen")),
    tooltip=["name", "weight", "horsepower"]
)


# If you want the colors to progress in the opposite order, you can add another keyword argument, `reverse=True`.  (This is getting added to the `alt.Scale` constructor.

# In[17]:


# syntax will seem weird at first
alt.Chart(df).mark_circle().encode(
    x="weight",
    y="horsepower",
    color=alt.Color("model_year", scale=alt.Scale(scheme="goldgreen", reverse=True)),
    tooltip=["name", "weight", "horsepower"]
)

