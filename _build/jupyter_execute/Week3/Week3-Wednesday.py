#!/usr/bin/env python
# coding: utf-8

# # Week 3 Wednesday

# ## Announcements
# 
# * I have office hours at 1pm today, downstairs in ALP 2800.  Please come by with any questions!
# * Videos and video quizzes for Friday are posted.

# ## Plan
# 
# * Encoding data types
# * Multi-view plots in altair
# * Time to work on Worksheets 5 and 6

# ## Encoding data types
# 
# (This notion of *quantitative* data vs *categorical* data will also be very important when we get to the Machine Learning portion of Math 10.)  Altair chooses different default values depending on the *type* of the data being encoded. These are the 5 types of data distinguished by Altair ([Reference](https://altair-viz.github.io/user_guide/encoding.html#encoding-data-types)):
# 
# | Data Type    | Shorthand Code | Description                       |
# |--------------|----------------|-----------------------------------|
# | quantitative | Q              | a continuous real-valued quantity |
# | ordinal      | O              | a discrete ordered quantity       |
# | nominal      | N              | a discrete unordered category     |
# | temporal     | T              | a time or date value              |
# | geojson      | G              | a geographic shape                |
# 
# For us, the most important are the first three, quantitative, ordinal, and nominal.  A **quantitative** data type is just an ordinary numeric data type, like floats.  **Ordinal** and **Nominal** data types are **categorical** data types, where the values represent discrete categories or classes.  We use the **ordinal** designation if the categories have a natural ordering and we use **nominal** if the categories do not have a natural ordering.

# * Load the "mpg" dataset (`sns.load_dataset`) from Seaborn and name the DataFrame `df`.
# * Find the sub-DataFrame for which the name of the car contains the substring "skylark".  Name the sub-DataFrame `df_sub`.  (You can use `map` and a lambda function, or you can use `str.contains`.  The `str.contains` approach is probably more elegant and more efficient, but the `map` approach is more general.)
# * Make a scatter plot in Altair from this sub-DataFrame using the "model_year" for both the x-coordinate and the color, and using "mpg" for the y-coordinate.  (We can increase the size of the points, and remove zero from the x-axis, to make it easier to see.)
# * What changes if you specify different encoding types for "model_year"?  (The difference in color between quantitative and ordinal will be more clear if you use a different color scheme, like "spectral".)

# In[1]:


import pandas as pd


# In[2]:


import seaborn as sns


# In[3]:


df = sns.load_dataset("mpg")


# I caught myself using the following "attribute notation" for accessing a column.  This is sometimes convenient, but it doesn't always work.  I'll try in general to use `df["name"]` instead of `df.name`, but it's good to know that they both exist.

# In[4]:


# abbreviation for df["name"].  The abbreviation doesn't always work
df.name


# Here is an example of the type of test we want to apply to every value.

# In[5]:


"skylark" in "buick skylark 320"


# We apply that test to every value using the panda Series `map` method.  The input to the `map` method needs to be a function.  In this case, we define that input function using a lambda function.

# In[6]:


df["name"].map(lambda s: "skylark" in s)


# Now that we have our Boolean Series, we can use Boolean indexing like usual.  This is a small DataFrame, with only 4 rows.

# In[7]:


df_sub = df[df["name"].map(lambda s: "skylark" in s)]
df_sub


# In[8]:


import altair as alt


# Here is how the chart looks with the default data types.  We are specifying a constant size (as opposed to encoding a column as the size), so we pass that constant size value to the `mark_circle` method.  The only reason we are increasing the size is so the points are easier to see.  By default, because the "model_year" column contains numbers, Altair assumes it is a *quantitative* data type.

# In[9]:


alt.Chart(df_sub).mark_circle(size=200).encode(
    x="model_year",
    y="mpg",
    color="model_year"
)


# Here we use the abbreviation `:N` to specify that the model year in the color encoding should be considered as a *nominal* data type.  Because nominal means it is unordered, Altair chooses colors that don't have any particular ordering (as opposed to above, where they progressed from light blue to dark blue).

# In[10]:


alt.Chart(df_sub).mark_circle(size=200).encode(
    x="model_year",
    y="mpg",
    color="model_year:N" # :N nominal
)


# Here we specify that the model year in the x-encoding is an ordinal data type.  Notice how the spacing between the years is no longer relevant; now we just get a single column per year, with an equal amount of space between them.

# In[11]:


alt.Chart(df_sub).mark_circle(size=200).encode(
    x="model_year:O",
    y="mpg",
    color="model_year:N" # :N nominal
)


# I hope the above examples give some sense for how some variables can be viewed as either quantitative or categorical.  This notion will be especially important when we get to machine learning, where quantitative variables correspond to *regression* problems, and where categorical variables correspond to *classification* problems.

# ## Multi-view plots in altair
# 
# Here we switch back to the full DataFrame, `df`.
# 
# * A *facet chart* breaks the dataset up into sub-datasets and makes a different chart for each sub-dataset.  Make a facet chart  using "horsepower" for the x-coordinate, "mpg" for the y-coordinate, "origin" for the color, and dividing the data according to the number of cylinders.  Put each chart in its own row.  [Reference](https://altair-viz.github.io/user_guide/compound_charts.html#faceted-charts) (scroll down to the `.facet` example).
# * The `facet` method is used to break the chart up into pieces, for example, all the cars with 4 cylinders all the cars with 6 cylinders, and so on.  Does that remind you of a pandas method?  (**We didn't answer this question.**)

# Let's first see how this `facet` method works on the example we were using above.

# In[23]:


df.cylinders.value_counts()


# When we eventually make our facet chart, one of the sub-charts will have 204 points, another will have 103 points, and so on.
# 
# Let's first see what the full data looks like from above.  By default, Altair includes zero in quantitative axes.  So that we get a more "zoomed in" view, we explicitly specify that zero should not be included, by specifying `scale=alt.Scale(zero=False)`.

# In[13]:


alt.Chart(df).mark_circle().encode(
    x=alt.X("model_year", scale=alt.Scale(zero=False)),
    y="mpg",
    color="model_year"
)


# There are 5 charts in the following, one chart for each possible "cylinders" value.

# In[14]:


alt.Chart(df).mark_circle().encode(
    x=alt.X("model_year", scale=alt.Scale(zero=False)),
    y="mpg",
    color="model_year"
).facet(
    row="cylinders"
)


# We used `row="cylinders"` to indicate that each cylinder value should appear in its own row.  If we just input `"cylinders"` without the `row=`, then each cylinder value will appear in its own column.

# In[15]:


alt.Chart(df).mark_circle().encode(
    x=alt.X("model_year", scale=alt.Scale(zero=False)),
    y="mpg",
    color="model_year" # :N nominal
).facet(
    "cylinders"
)


# Let's now consider the actual example I asked for in the first bullet point above.  Notice how this presentation clearly shows the trend for cars with more cylinders to have higher horsepower (because as we move down through the charts, the points move to the right).

# In[16]:


alt.Chart(df).mark_circle().encode(
    x=alt.X("horsepower", scale=alt.Scale(zero=False)),
    y="mpg",
    color="origin" # :N nominal
).facet(
    row="cylinders"
)

