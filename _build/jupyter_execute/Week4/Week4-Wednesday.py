#!/usr/bin/env python
# coding: utf-8

# # Week 4 Wednesday

# ## Announcements
# 
# * I have office hours at 1pm, downstairs in ALP 2800.  Please come by with any questions!
# * Videos and video quizzes posted; due Friday before lecture.  (I plan to close all the quizzes and convert them to "practice quizzes" sometime before the midterm, so you can use them to study.  If you are behind on the video quizzes, this is a good time to catch up.)
# * Midterm next Wednesday, October 26th, during lecture.  Similar question style to the in-class quizzes, but some questions may be longer.  Sample midterm posted by the end of this week.
# * Note cards will be passed out later today during the "worksheet time".  (Remind me if I forget.)  You can put hand-written notes on them (both sides, one card per student) and use them during the midterm.

# ## A DataFrame which is difficult to use "as is" with Altair
# 
# * Using the following pandas DataFrame, draw a line that goes 60, 80, 65 for Irvine and another line that goes 25, 85, 50 for New York.  Why is this more difficult than you would expect?

# In[1]:


import pandas as pd
import altair as alt


# Here is a very simple DataFrame, but its data is presented slightly differently from how Altair expects.

# In[2]:


df = pd.DataFrame({
    "City": ["Irvine", "New York"],
    "Feb": [60, 25],
    "Jul": [80, 85],
    "Nov": [65, 50]
})


# Because there is no single column containing the month values, we do not currently know how to plot those month values along an axis using Altair.  It would be easy to plot, for example "Irvine" and "New York" along the x-axis, or to plot 25 and 60 along the x-axis, but not to plot "Feb", "Jul", "Nov" along the x-axis.

# In[3]:


df


# The solution is to use the pandas DataFrame method `melt`.

# In[4]:


df.melt(
    id_vars=["City"], # columns to keep the same
    var_name="Month", # the other column labels go here
    value_name="Temperature", # the old values go here
    )


# The syntax takes some getting used to.  It can seem like magic that the month labels and temperatures showed up in the correct spot.  Here is another example, where we specify to leave both the "City" and the "Jul" columns unchanged.

# In[5]:


df.melt(
    id_vars=["City", "Jul"], # columns to keep the same
    var_name="Month", # the other column labels go here
    value_name="Temperature", # the old values go here
    )


# pandas did not know that the "Feb" corresponded to a "Month"... we told it that.  If we chose different names, then the newly formed columns would have different labels.  Notice how `var_name` describes what to call the old column labels, and `value_name` describes what to call the old values in those columns.

# In[6]:


df.melt(
    id_vars=["City"], # columns to keep the same
    var_name="Variable", # the other column labels go here
    value_name="Value", # the old values go here
    )


# A common source of mistakes in Python is thinking that code like the following changed `df`.  A hint that `df` did not change is the fact that a new DataFrame got displayed.

# In[7]:


# this code does not change df
df.melt(
    id_vars=["City"], # columns to keep the same
    var_name="Month", # the other column labels go here
    value_name="Temperature", # the old values go here
    )


# Nothing we have done so far has changed `df`.

# In[8]:


df


# Here we store the melted DataFrame in a new variable name `df2`.  Notice how nothing gets displayed beneath this cell.

# In[9]:


df2 = df.melt(
    id_vars=["City"], # columns to keep the same
    var_name="Month", # the other column labels go here
    value_name="Temperature", # the old values go here
    )


# This is almost what we want, but we haven't told Altair to do anything with the "City" column yet.

# In[11]:


alt.Chart(df2).mark_line().encode(
    x="Month",
    y="Temperature",
)


# This is the kind of chart we were looking for.  You will need to do something similar on Worksheet 8, where we are displaying various assignment names along the x-axis, like "Quiz 1" and "Quiz 2".

# In[12]:


alt.Chart(df2).mark_line().encode(
    x="Month",
    y="Temperature",
    color="City"
)


# ## Interactive chart, example 1
# 
# * Run `alt.data_transformers.enable('default', max_rows=10000)` so you can plot points from up to 10,000 rows in a DataFrame.  (**Warning**.  Don't use numbers much higher than this.  Because every data point is plotted, the file sizes can become huge.)
# * Using the normalized stock data from Worksheet 4 (attached), make a line chart which highlights a certain stock market when you click on the legend.

# In[13]:


df = pd.read_csv("wk4.csv")


# In[14]:


df.shape


# In[15]:


df.columns


# Here is the error Altair will raise if you try to plot from a DataFrame with more than 5000 rows.

# In[16]:


alt.Chart(df).mark_line().encode(
    x="Date",
    y="NormOpen",
    color="Abbreviation"
)


# Here we specify that Altair should allow up to 10,000 rows.  Be careful with this tool; I do not think you should allow more than maybe 20,000 rows.  The risk is producing a huge file, and possibly crashing the machine.

# In[17]:


alt.data_transformers.enable('default', max_rows=10000)


# Because `df` had over 60,000 rows, we still need to decrease the size of `df` somehow.  Here we use `sample` to get only 30 rows.

# In[19]:


# still need to shrink df
alt.Chart(df.sample(30)).mark_line().encode(
    x="Date",
    y="NormOpen",
    color="Abbreviation"
)


# By default, Altair doesn't know that the "Date" column holds values representing dates.  We can tell Altair this by specifying `:T` as the encoding data type.  (Another option would be to use `pd.to_datetime` on the "Date" column, and then Altair would recognize automatically that these represent datetime values.)  If you try to plot 10,000 points using just string encodings, the file will be huge and it will probably not be displayed.

# In[18]:


# still need to shrink df
alt.Chart(df.sample(10000)).mark_line().encode(
    x="Date:T",
    y="NormOpen",
    color="Abbreviation"
)


# Now we finally get to interactivity.
# 
# Step 1.  Create an Altair selection object.  Here we specify that we want to select objects by the "Abbreviation" field.

# In[20]:


sel = alt.selection_single(fields=["Abbreviation"], bind="legend")

alt.Chart(df.sample(10000)).mark_line().encode(
    x="Date:T",
    y="NormOpen",
    color="Abbreviation"
)


# Step 2.  Add the selection object to the chart using `add_selection`.

# In[22]:


sel = alt.selection_single(fields=["Abbreviation"], bind="legend")

alt.Chart(df.sample(10000)).mark_line().encode(
    x="Date:T",
    y="NormOpen",
    color="Abbreviation"
).add_selection(sel)


# Step 3.  Tell Altair how to respond to the selection.  Here we use `alt.condition` to say that if the point is selected, use the default coloring and an opacity of `1`, and if the point is not selected, use light grey for the color and make the line 80% transparent (an opacity of `0.2`).
# 
# Try clicking on one of the stock exchange abbreviations listed in the legend below.  Notice how the chart responds.

# In[25]:


sel = alt.selection_single(fields=["Abbreviation"], bind="legend")

alt.Chart(df.sample(10000)).mark_line().encode(
    x="Date:T",
    y="NormOpen",
    color=alt.condition(sel, "Abbreviation", alt.value("lightgrey")),
    opacity=alt.condition(sel, alt.value(1), alt.value(0.2))
).add_selection(sel)


# You will see more examples of interactivity on Worksheet 8.  A very nice aspect of this interactivity is that, once the visualization is produced, the interactivity can be presented on any website, even if Python is not available to the website.

# ## Interactive chart, example 2
# 
# * Using the "mpg" dataset from Seaborn, make a scatter plot showing "horsepower" vs "mpg" together with make a bar chart that shows how many cars there are from each origin.
# 
# We didn't get here on Wednesday.
