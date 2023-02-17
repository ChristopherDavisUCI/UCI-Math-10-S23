#!/usr/bin/env python
# coding: utf-8

# # Week 4 Friday

# ## Announcements
# 
# * Video quizzes due.  (You can still get credit on all the video quizzes up until Sunday, so if you're behind, try to catch up this weekend.  Early Monday I will convert them to practice quizzes so you can use them to practice for the midterm.)
# * Midterm Wednesday.  Covers material up to and including anything covered Monday.
# * Pick up a notecard if you haven't already.  Hand-written notes, written on both sides, can be used during the midterm.
# * Sample midterm posted on the "Week 5" page on Canvas.  I'll ask William to go over as much of the sample midterm as possible on Tuesday.

# In[2]:


import pandas as pd
import altair as alt
import seaborn as sns


# Just for a change of pace, we are calling the DataFrame `cars` instead of the usual `df` today.

# In[3]:


cars = sns.load_dataset("mpg")


# ## Interactive bar chart
# 
# * Using the "mpg" dataset from Seaborn, display a scatterplot of horsepower vs mpg, colored by origin, alongside a bar chart, showing how many cars there are from each origin.

# We need to use `cars` instead of `df` when we specify the pandas DataFrame for this Altair chart.

# In[3]:


alt.Chart(df).mark_circle().encode(
    x="horsepower",
    y="mpg",
    color="origin"
)


# Here is the basic scatter plot for this data.

# In[4]:


c1 = alt.Chart(cars).mark_circle().encode(
    x="horsepower",
    y="mpg",
    color="origin"
)

c1


# Here is the basic bar chart for this data.  Notice how we use `"count()"` to tell Altair to count how many instances there are in each category.  For example, the following says that there are about 70 cars from Europe in this dataset.

# In[5]:


c2 = alt.Chart(cars).mark_bar().encode(
    x="origin",
    y="count()",
    color="origin"
)

c2


# Similar to `count`, there are other functions we can use, like `sum` and `mean` and `median`.  Using these on their own does not make sense, like `y="sum()"` would not make sense... instead, we need to specify what should be summed.  Here we are summing the "horsepower" column.  (If there were a column that contained all 1s, we could sum that column to get the same thing as `count()`.)

# In[6]:


# what about sum?

alt.Chart(cars).mark_bar().encode(
    x="origin",
    y="sum(horsepower)",
    color="origin"
)


# * Sample question: Try to find the heights of these blue bars exactly using pandas and Boolean indexing.

# In[8]:


df_europe = cars[cars["origin"] == "europe"]


# Here is one way to find how many rows there are in `df_europe`.  Notice how this matches the height of our bar above.

# In[9]:


# count
df_europe.shape[0]


# Here is another approach, using the built-in Python function `len` to get the length of the DataFrame.

# In[10]:


len(df_europe)


# Notice how the following matches our `y="sum(horsepower)"` height for Europe.

# In[11]:


df_europe["horsepower"].sum()


# I expected the following to work but to possibly be slower.  Instead, it doesn't work because of missing values (or rather, it works, but it does not give us any information about the height of the bar).

# In[12]:


# might be slower since it's not specific to pandas
# also handles missing values differently
sum(df_europe["horsepower"])


# Back to our original charts.

# In[7]:


c1


# In[8]:


c2


# Here we create a selection object (step 1) and add it to `c1` (step 2).  Now you can click and drag and a grey rectangle shows up, but there is no responsiveness yet.  (On Wednesday, we used `alt.selection_single` to choose one thing at a time; here, we are using `alt.selection_interval` to choose an interval at a time.)

# In[9]:


brush = alt.selection_interval() # step 1

c1 = alt.Chart(cars).mark_circle().encode(
    x="horsepower",
    y="mpg",
    color="origin"
).add_selection(brush) # step 2

c2 = alt.Chart(cars).mark_bar().encode(
    x="origin",
    y="count()",
    color="origin"
)

alt.hconcat(c1,c2) # c1|c2


# * Using an Altair selection interval object along with `transform_filter`, change the heights of the bars to match how many cars were selected.  Use the `scale` keyword argument so that the y-axis of the bar chart does not change.

# Here we add the responsiveness, in the `transform_filter(brush)` portion.  This tells Altair to only use objects which have been selected.  We also specify a fixed `domain` for the y-axis so that the height of the bars are easier to compare.
# 
# Try clicking and dragging on the left chart, and notice how the bar heights respond.

# In[10]:


brush = alt.selection_interval() # step 1

c1 = alt.Chart(cars).mark_circle().encode(
    x="horsepower",
    y="mpg",
    color="origin"
).add_selection(brush) # step 2

c2 = alt.Chart(cars).mark_bar().encode(
    x="origin",
    y=alt.Y("count()", scale=alt.Scale(domain=[0,260])),
    color="origin"
).transform_filter(brush)

alt.hconcat(c1,c2) # c1|c2


# Because `c2` references `brush`, but we never added `brush` to `c2`, if we try to view `c2` by itself, we get an error.  (We didn't get an error above, because we had it displayed together with `c1` using `alt.hconcat`.)

# In[11]:


c2


# The chart `c1` does work on its own, there's just no responsiveness.

# In[12]:


c1


# I'm not sure why the following doesn't work.  I assumed that when we clicked and dragged on it, all the non-selected points would disappear.

# In[13]:


# Not sure why this doesn't work
c1.transform_filter(brush)


# * As review from Wednesday, change the colors in the scatterplot so that unselected points get a constant color, using `alt.value`.  (Aside about colors.  You can use any valid CSS color string.  There are [very many options](https://developer.mozilla.org/en-US/docs/Web/CSS/color_value) for these CSS color strings.  The easiest option is to use a [named color](https://developer.mozilla.org/en-US/docs/Web/CSS/named-color).  Another common option is to specify [RGB values](https://developer.mozilla.org/en-US/docs/Web/CSS/color_value/rgb).  I haven't learned it myself yet, but I believe [HSL values](https://developer.mozilla.org/en-US/docs/Web/CSS/color_value/hsl) in general are supposed to be easier to work with than RGB values, in terms of the colors produced.  I heard this article [Color for the Color-challenged](https://ferdychristant.com/color-for-the-color-challenged-884c7aa04a56) recommended on an episode of the podcast [syntax.fm](https://syntax.fm/show/479/css5-color-functions).)

# Notice how if you click and drag, the unselected points now have an "orchid" color.  (Notice how we need to use `alt.value("orchid")`.  If we just used `"orchid"`, Altair would be looking for a column in the DataFrame named "orchid".)

# In[14]:


brush = alt.selection_interval() # step 1

c1 = alt.Chart(cars).mark_circle().encode(
    x="horsepower",
    y="mpg",
    color=alt.condition(brush, "origin", alt.value("orchid"))
).add_selection(brush) # step 2

c2 = alt.Chart(cars).mark_bar().encode(
    x="origin",
    y=alt.Y("count()", scale=alt.Scale(domain=[0,260])),
    color="origin"
).transform_filter(brush)

alt.hconcat(c1,c2) # c1|c2


# Here is an example of using a different color encoding.  Selected points are colored according to "horsepower"; unselected points are still "orchid".

# In[15]:


brush = alt.selection_interval() # step 1

c1 = alt.Chart(cars).mark_circle().encode(
    x="horsepower",
    y="mpg",
    color=alt.condition(brush, "horsepower", alt.value("orchid"))
).add_selection(brush) # step 2

c2 = alt.Chart(cars).mark_bar().encode(
    x="origin",
    y=alt.Y("count()", scale=alt.Scale(domain=[0,260])),
    color="origin"
).transform_filter(brush)

alt.hconcat(c1,c2) # c1|c2


# Here is one more example of how `alt.condition` works.  Selected points are colored black; unselected points are colored orchid.

# In[16]:


brush = alt.selection_interval() # step 1

c1 = alt.Chart(cars).mark_circle().encode(
    x="horsepower",
    y="mpg",
    color=alt.condition(brush, alt.value("black"), alt.value("orchid"))
).add_selection(brush) # step 2

c2 = alt.Chart(cars).mark_bar().encode(
    x="origin",
    y=alt.Y("count()", scale=alt.Scale(domain=[0,260])),
    color="origin"
).transform_filter(brush)

alt.hconcat(c1,c2) # c1|c2


# * Specify that the selection interval should only be bound to the x-encoding, and specify an initial value with the `init` keyword argument.  The `init` keyword argument should be assigned to a dictionary, which in this case will have only one key.  Can you guess what the key-value pair should look like?

# Notice how now we have no control over the vertical height of the selection interval; we can only control the span of the x encoding.

# In[17]:


brush = alt.selection_interval(encodings=["x"]) # step 1

c1 = alt.Chart(cars).mark_circle().encode(
    x="horsepower",
    y="mpg",
    color=alt.condition(brush, "origin", alt.value("orchid"))
).add_selection(brush) # step 2

c2 = alt.Chart(cars).mark_bar().encode(
    x="origin",
    y=alt.Y("count()", scale=alt.Scale(domain=[0,260])),
    color="origin"
).transform_filter(brush)

alt.hconcat(c1,c2) # c1|c2


# Specifying something like `{"x": 100}` wouldn't really make sense, because to specify an interval, we need to specify both endpoints of the interval.  So we specify `init={"x": [100,140]}`.  Notice how the chart starts with this interval selected.

# In[18]:


brush = alt.selection_interval(encodings=["x"], init={"x": [100,140]}) # step 1

c1 = alt.Chart(cars).mark_circle().encode(
    x="horsepower",
    y="mpg",
    color=alt.condition(brush, "origin", alt.value("orchid"))
).add_selection(brush) # step 2

c2 = alt.Chart(cars).mark_bar().encode(
    x="origin",
    y=alt.Y("count()", scale=alt.Scale(domain=[0,260])),
    color="origin"
).transform_filter(brush)

alt.hconcat(c1,c2) # c1|c2


# ## Review
# 
# I'm not sure how much time will be left today.  Be thinking about what topics you would like to review.  (Please phrase them in terms of the Python topic, as opposed to for example "Worksheet 5, #4".)

# We'll talk more about `apply` on Monday.  (Secretly there is also an `apply` method for pandas Series, but we will never use that in Math 10.)

# * map: is a Series method
# * apply and applymap: DataFrame methods

# In[19]:


df = pd.DataFrame({
    "A": [3,1,4,1],
    "B": [2,0,2,5]
})


# If the following is confusing, try breaking it up into smaller pieces, like `df["A"]`, then `df["A"].map(lambda x: x**2)`, then `df["A"].map(lambda x: x**2 > 10)`, then `df["A"].map(lambda x: x**2 > 10).sum()`.

# In[20]:


df["A"].map(lambda x: x**2 > 10).sum()


# Here is an example of applying the function `str` to every entry in `df`.  

# In[21]:


# df.applymap(f) applies f to every entry in df
df2 = df.applymap(str)


# The DataFrame looks the same, but the entries have been converted to strings.

# In[22]:


df2


# Let's check that the lower-right `5` is really a string.

# In[36]:


type(df2.loc[3,"B"])

