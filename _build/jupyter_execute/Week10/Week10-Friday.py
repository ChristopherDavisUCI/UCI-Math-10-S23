#!/usr/bin/env python
# coding: utf-8

# # Week 10 Friday
# 
# I've attached a dataset containing hourly temperatures from Kaggle: [source](https://www.kaggle.com/datasets/selfishgene/historical-hourly-weather-data)
# 
# The unit for these temperatures is kelvin.
# 
# The idea for using a decision tree for this kind of data set comes from Jake VanderPlas's [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/) (which was also the source of the bicycle worksheet).

# ## Announcements
# 
# * Be sure you're writing your project using the **Project Template** in the Project folder.  (This is different from the Worksheet 16 template.)
# * I have Zoom office hours Monday during the first hour of the time scheduled for our final exam: Monday, 10:30-11:30am.  See our Canvas homepage for the Zoom link.
# * Projects due 11:59pm Monday.  (An extension of 1-2 days may be possible on an individual basis; email me Sunday or Monday if you think you need one.  Share what you have so far.)
# * If you get stuck on something or have a logistical question, please ask on Ed Discussion, even if you think it is very specific to your own project.
# * Please fill out a [course evaluation](https://evaluations.eee.uci.edu/assigned/action_list) if you haven't already! 

# ## A decision tree to predict temperature
# 
# We won't divide into a training set and a test set in this case.  My intuition is that because the temperature at 3pm (say) is so similar to the temperature at 4pm, randomly dividing the data won't be appropriate.  Phrased another way, our eventual goal is probably to predict the temperature at some future date, not at some random hour in the middle of a day where we already know other temperatures.
# 
# My goal here is to see what the decision tree prediction function looks like.  I'm not thinking about overfitting right now.

# In[2]:


import pandas as pd
import altair as alt


# In[3]:


df_pre = pd.read_csv("temperature.csv")


# Different cities are in different columns.  If we wanted for example to plot these temperatures in different colors, we should use `df_pre.melt` to get the city columns combined into a single column.

# In[4]:


df_pre


# We will just take approximately 200 of the rows and two of the columns from this dataset.

# In[6]:


# greatly reduce the rows and columns
df = df_pre.loc[400:600, ["datetime", "Detroit"]].copy()


# In[7]:


df


# Notice how strange the x-axis labels look.  That is a sign that something is wrong with the data type of the "datetime" column.

# In[8]:


c1 = alt.Chart(df).mark_line().encode(
    x="datetime",
    y=alt.Y("Detroit", scale=alt.Scale(zero=False), title="kelvin")
).properties(
    width=700,
    title="Detroit"
)

c1


# These values are strings, not datetime objects.

# In[9]:


df.dtypes


# We convert that column into the datetime data type.  We could replace the old column, but here we include it as a new column.  (That way, if we make a mistake in this cell, we don't need to re-load the data.)

# In[10]:


df["date"] = pd.to_datetime(df["datetime"])


# In[11]:


df.dtypes


# Now the image looks much more natural.

# In[12]:


c1 = alt.Chart(df).mark_line().encode(
    x="date",
    y=alt.Y("Detroit", scale=alt.Scale(zero=False), title="kelvin")
).properties(
    width=700,
    title="Detroit"
)

c1


# It would be incorrect to use a classifier in this context, because predicting temperature is a regression problem.

# In[13]:


from sklearn.tree import DecisionTreeClassifier


# Here is the correct import.

# In[14]:


from sklearn.tree import DecisionTreeRegressor


# We'll specify 15 leaf nodes when we instantiate the regressor object.

# In[15]:


reg = DecisionTreeRegressor(max_leaf_nodes=15)


# Here is our usual error.  Remember that the first input needs to be two-dimensional.

# In[16]:


reg.fit(df["date"], df["Detroit"])


# In[17]:


reg.fit(df[["date"]], df["Detroit"])


# We add the predictions as a new column.  I expect this would raise a warning if we hadn't used `copy()` above.

# In[18]:


df["pred_tree"] = reg.predict(df[["date"]])


# Here are the predictions.  If you count, there should be 15 horizontal line segments, corresponding to the 15 leaf nodes.  (Remember that each leaf node corresponds to a region, and all inputs in a region have the same output.)

# In[19]:


c1 = alt.Chart(df).mark_line().encode(
    x="date",
    y=alt.Y("Detroit", scale=alt.Scale(zero=False), title="kelvin")
).properties(
    width=700,
    title="Detroit"
)

c2 = alt.Chart(df).mark_line(color="orange").encode(
    x="date",
    y=alt.Y("pred_tree", scale=alt.Scale(zero=False), title="kelvin")
    # Doesn't work.  color="orange"
    # Does work.  color=alt.value("orange")
)

c1+c2


# This setup is a little different from our other decision tree examples.  Notice how there is only one input column.  So for example, the feature importances array is not interesting in this case, because there is only one feature.

# In[20]:


reg.feature_importances_


# Let's see how this compares if we use a random forest with 100 trees, each with at most 15 leaf nodes.

# In[21]:


from sklearn.ensemble import RandomForestRegressor


# In[22]:


rfe = RandomForestRegressor(n_estimators=100, max_leaf_nodes=15)


# In[23]:


rfe.fit(df[["date"]], df["Detroit"])


# In[24]:


df["pred_forest"] = rfe.predict(df[["date"]])


# The image looks pretty similar in this case, but the corners are rounded because of the averaging that happens.  Look for example on the very right side of this picture, and compare it to the picture above.  We have a diagonal curve in this random forest plot, which is very different from the straight horizontal segments of the decision tree plot.

# In[25]:


c1 = alt.Chart(df).mark_line().encode(
    x="date",
    y=alt.Y("Detroit", scale=alt.Scale(zero=False), title="kelvin")
).properties(
    width=700,
    title="Detroit"
)

c2 = alt.Chart(df).mark_line(color="orange").encode(
    x="date",
    y=alt.Y("pred_forest", scale=alt.Scale(zero=False), title="kelvin")
    # Doesn't work.  color="orange"
    # Does work.  color=alt.value("orange")
)

c1+c2


# We could definitely keep going, for example considering overfitting or finding changes to make to get the plot to more closely match the true data, but this is where we'll finish the course!
