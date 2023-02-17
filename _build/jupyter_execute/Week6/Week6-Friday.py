#!/usr/bin/env python
# coding: utf-8

# # Week 6 Friday

# ## Announcements
# 
# * Videos and video quizzes due.
# * Worksheets 9 and 10 due Tuesday.
# * In-class quiz Tuesday based on K-means clustering and also has one question on `StandardScaler` (William covered on Tuesday).
# 
# The goal today is to see some aspects of linear regression using the "mpg" dataset from Seaborn.  I assume we won't get through all the material listed below.

# ## Linear Regression with one input variable
# 
# Find the line of best fit using the mpg dataset from Seaborn to model "mpg" as a function of the one input variable "horsepower".  The input variables are often called *features* or *predictors* and the output variable is often called the *target*.

# In[1]:


import pandas as pd
import numpy as np

import altair as alt
import seaborn as sns


# We will get errors using scikit-learn if there are missing values (at least without some extra arguments), so here we drop all the rows which have missing values.

# In[2]:


df = sns.load_dataset("mpg").dropna(axis=0)


# Notice how the following chart shows (matching our intuition) that as horsepower increases, mpg decreases.

# In[3]:


base = alt.Chart(df).mark_circle().encode(
    x="horsepower",
    y="mpg"
)

base


# Let's see the same thing using scikit-learn's LinearRegression class.  Linear regression is an example of *supervised* machine learning (as opposed to unsupervised machine learning, like the clustering we were doing before).  The fact that it is supervised machine learning, means that we need to have answers for at least some of our data.  (In this case, we have answers, i.e., we have the true "mpg" value, for all of the data.)

# In[4]:


from sklearn.linear_model import LinearRegression


# In[5]:


reg = LinearRegression()


# In[6]:


type(reg)


# Here is one of the most common errors when using scikit-learn.  It wants the input to be two-dimensional, even if it's just a single column in a DataFrame.  (The reason is that, when there are multiple input columns, the input needs to be two-dimensional, so it's easier for scikit-learn if the inputs are always two-dimensional.

# In[7]:


reg.fit(df["horsepower"], df["mpg"])


# Here is the input we tried to use.

# In[10]:


df["horsepower"]


# Here is the input we are going to use.  It looks very similar, but because it is a pandas DataFrame (instead of a pandas Series), scikit-learn knows how to work with it as an input.

# In[11]:


df[["horsepower"]]


# In[12]:


type(df[["horsepower"]])


# Notice how the output (also called the *target*) `df["mpg"]` remains one-dimensional.

# In[9]:


reg.fit(df[["horsepower"]], df["mpg"])


# We can now make predictions for the miles per gallon using the "horsepower" column.

# In[12]:


df["pred1"] = reg.predict(df[["horsepower"]])


# Notice how our DataFrame now includes both the "mpg" column (the true values) as well as the "pred1" column all the way on the right (the predicted values).

# In[13]:


df.head()


# Let's see how these predicted values look.

# In[14]:


c1 = alt.Chart(df).mark_line().encode(
    x="horsepower",
    y="pred1"
)


# The line on the following chart should be considered the "line of best fit" modeling miles-per-gallon as a linear function of horsepower.

# In[15]:


base+c1 # alt.layer(base, c1)


# If you look at the line, the following value of the y-intercept should be believable.

# In[17]:


reg.intercept_


# Notice how the coefficient is negative.  This corresponds to the line having negative slope, and it matches our intuition that, as "horsepower" increases, "mpg" decreases.  (The number is shown as a length-1 NumPy array because, typically we will be using multiple input columns, and an array will be used to store all of the coefficients together.)

# In[16]:


reg.coef_


# ## Linear Regression with multiple input variables
# 
# Now model "mpg" as a function of the following input variables/predictors/features:
# ```
# ["horsepower", "weight", "model_year", "cylinders"]
# ```

# The routine is very similar, just using multiple input columns (four in this case).

# In[18]:


reg2 = LinearRegression()


# In[19]:


cols = ["horsepower", "weight", "model_year", "cylinders"]


# Notice how we write `df[cols]` here and we wrote `df[["horsepower"]]` above.  This might seem contradictory, but these are analogues of each other, because `cols` is a list and `["horsepower"]` is also a list (a length one list).

# In[20]:


reg2.fit(df[cols], df["mpg"])


# Here are the coefficients, stored in a NumPy array.  There are four of these numbers because we used four columns.

# In[21]:


reg2.coef_


# We want to know which coefficient corresponds to which column (otherwise the numbers are not very meaningful).  We could look back at the `cols` list, but we can also get the same information from `reg2` using its `feature_names_in_` attribute.
# 
# One of the best features of linear regression is that it is very possible to interpret the values it produces.  For example, the `-0.0036` above should be interpreted as the partial derivative of "mpg" with respect to "horsepower" for our linear model.  Notice that most of these are negative, but the coefficient for "model_year" is positive.  It makes sense that cars tend to have higher mpg values as the model year increases.

# In[24]:


reg2.feature_names_in_


# Again we can make predictions using this data.

# In[22]:


df["pred2"] = reg2.predict(df[cols])


# In[23]:


c2 = alt.Chart(df).mark_line().encode(
    x="horsepower",
    y="pred2"
)


# This chart looks pretty crazy.  We will describe it more in the same cell.

# In[24]:


base+c2


# The predicted values do not look very linear, but that is because the line chart comes from data points which have four different values associated to them: "horsepower", "weight", "model_year", "cylinders".  Our x-axis only shows "horsepower", but the points on the line depend on all four values.
# 
# In the following cell, we add a tooltip to the base scatterplot.  Put your mouse over the low point near horsepower 130 and over the highest point near horsepower 130.  Even though these two points have roughly the same horsepower (130 and 132), the weights are very different (3870 and 2910, respectively), so that is why our line chart includes a lower miles per gallon point (for the higher weight) and a higher miles per gallon point.
# 
# This is a confusing point.  You will get a chance to think about something similar to it on one of next week's homeworks.

# In[25]:


base = alt.Chart(df).mark_circle().encode(
    x="horsepower",
    y="mpg",
    tooltip=cols
)

base+c2


# ## Linear Regression using rescaled features
# 
# Use a `StandardScaler` object to rescale these four input features, and then perform the same linear regression.

# We are going to change some of the values in the DataFrame (by rescaling them), so it seems safest to first make a copy of the DataFrame.

# In[31]:


df2 = df.copy()


# I believe William introduced this `StandardScaler` class on Tuesday.

# In[32]:


from sklearn.preprocessing import StandardScaler


# In[33]:


scaler = StandardScaler()


# The syntax is the same as for `KMeans` and `LinearRegression`

# In[34]:


scaler.fit(df[cols])


# One difference is that we use `transform` instead of `predict`.  That is because we are not predicting anything. 

# In[35]:


df2[cols] = scaler.transform(df[cols])


# Notice how the four columns "horsepower", "weight", "model_year", "cylinders" have changed dramatically.

# In[36]:


df2


# Those four columns now have `mean` very close to 0.  (For K-means clustering, there is no need to change the mean, because we are subtracting one from the other, so any shift by a constant amount will disappear.)

# (I didn't see any warnings on Deepnote, but the following raises warnings about using numeric columns only.)

# In[39]:


df2.mean(axis=0)


# Notice how the standard deviations are close to 1.  (As far as I know, there is no specific meaning to the actual numbers that show up.  All that's important are that they are close to 1.)

# In[40]:


df2.std(axis=0)


# We can now perform the same procedure as above.

# In[41]:


reg3 = LinearRegression()


# In[42]:


reg3.fit(df2[cols], df2["mpg"])


# In[43]:


reg3.feature_names_in_


# The relative magnitudes of the coefficients in `reg2` were not meaningful (as far as I know), because the scales of the input features were different (in fact, they all had different units).  By rescaling the data, the following magnitudes become meaningful.  For example, because the scaled "weight" coefficient has the biggest absolute value, it should be interpreted as the most important of these four features with respect to mpg.  That was not at all obvious from the numbers we saw above.

# In[44]:


reg3.coef_


# Here is an elegant way to group those numbers and feature names together into a pandas Series.  As a first step, we can make a pandas Series.

# In[41]:


pd.Series(reg3.coef_)


# Here is how we can assign names to each of the numbers.

# In[42]:


pd.Series(reg3.coef_, index=reg3.feature_names_in_)


# Here we sort the pandas Series, with the biggest values at the beginning.  For this sorting, we only care about the size of the absolute value of the numbers; that is why we use `key=abs` in the following `sort_values` pandas Series method.

# In[45]:


pd.Series(reg3.coef_, index=reg3.feature_names_in_).sort_values(ascending=False, key=abs)


# ## Linear Regression using a categorical variable
# 
# * Again perform linear regression, this time also including "origin" as a predictor.  Use a `OneHotEncoder` object.
# * Remove the `intercept` (also called bias) when we instantiate the `LinearRegression` object.
# 
# (Aside.  It's not obvious to me whether we should rescale this new categorical feature.  For now we won't rescale it.  It's also not obvious to me if we should rescale the output variable.  Some quick Google searches suggest there are pros and cons to both.)
