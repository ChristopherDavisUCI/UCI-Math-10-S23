#!/usr/bin/env python
# coding: utf-8

# # Week 7 Monday

# ## Announcements
# 
# * The next videos and video quizzes are posted.  Due Monday of Week 8 (because Friday is a holiday this week).
# * Worksheets 9 and 10 due tomorrow.
# * In-class quiz tomorrow based on K-means clustering and also has one question on `StandardScaler` (which William discussed on Tuesday and we discussed on Friday).
# * I have office hours after class at 11am, next door in ALP 3610.
# * I posted sample solutions to Midterm 1.  They're linked from the Course Updates section on our Canvas homepage.

# In[1]:


import pandas as pd


# ## The pandas DataFrame `merge` method
# 
# The `merge` method could have been covered in the first half of Math 10.  We are covering it now because it appears on today's worksheet. There is also a related method `join`, but I am in the habit of using `merge`.
# 
# * Combine the data from the following two DataFrames using `df_menu.merge` together with `how="inner"`.
# * Practice question (won't go over today): can you make the same result using the `map` method, like what we did with the stocks dataset?

# In[2]:


df_menu = pd.DataFrame({
                    "item": ["BLT", "Egg sandwich", "Ramen", "Pumpkin Pie",
                            "Caesar salad", "House salad", "Falafel wrap"],
                    "food_type": ["Sandwich", "Sandwich", "Soup", "Dessert",
                            "Salad", "Salad", "Sandwich"],
                    "Vegan": [False, False, False, False, False, True, True]
                    })

df_price = pd.DataFrame({
                    "food_type": ["Drink", "Salad", "Sandwich", "Soup"],
                    "price": [5, 16, 12, 10]
                    })


# In[3]:


df_menu


# In[4]:


df_price


# The following doesn't raise an error, but it also doesn't do what we want.  For example, the prices are not aligned with the correct categories.

# In[5]:


# doesn't work
pd.concat((df_menu, df_price), axis=1)


# The following `how="left"` syntax and its variants are related to database methods.  In this case, because the `df_menu` DataFrame is the DataFrame that occurs on the left side of the code, we are telling pandas to keep all of the rows from that DataFrame.  We match them to rows in the other DataFrame using the `"food_type"` column.

# In[6]:


df_menu.merge(df_price, on="food_type", how="left")


# The following is similar, but we use all of the rows from the `df_price` DataFrame.  Notice how the "Salad" row occurs twice, because it occurs twice in the `df_menu` DataFrame.  Notice how the "Pumpkin Pie" row does not show up, because "Dessert" does not occur in the right DataFrame.  (Don't worry about the order of the rows.  I'm not certain how that is determined.)

# In[7]:


df_menu.merge(df_price, on="food_type", how="right")


# The keyword argument `how="inner"` is like an intersection.  We only keep rows if the corresponding "food_type" value occurs in both DataFrames.  Notice how there are no missing values in the following DataFrame, unlike the two previous DataFrames which did have `nan` values.

# In[8]:


df_menu.merge(df_price, on="food_type", how="inner")


# The following is like a union.

# In[9]:


df_menu.merge(df_price, on="food_type", how="outer")


# What if we don't specify the `how` keyword argument?  We can check the documentation, which shows us that `"inner"` is the default value of this keyword argument.  So if we don't specify `how`, it will be the same as specifying `how="inner"`.

# In[10]:


help(df_menu.merge)


# ## Some code from the last class
# 
# Here is the base code we were using for linear regression on Friday.

# In[11]:


import pandas as pd
import numpy as np

import altair as alt
import seaborn as sns

from sklearn.linear_model import LinearRegression


# In[12]:


df = sns.load_dataset("mpg").dropna(axis=0)

cols = ["horsepower", "weight", "model_year", "cylinders"]


# In[13]:


reg = LinearRegression()


# In[14]:


reg.fit(df[cols], df["mpg"])


# In[15]:


pd.Series(reg.coef_, index=reg.feature_names_in_)


# In[16]:


df.head(3)


# ## Linear Regression using a categorical variable
# 
# * Again perform linear regression, this time also including "origin" as an additional predictor.  Use a `OneHotEncoder` object.
# * These three new origin columns act like a separate intercept for each origin (one intercept for "europe", one intercept for "japan", one intercept for "usa"). Thus it makes sense to remove the `intercept` (also called bias) when we instantiate the `LinearRegression` object.

# In[17]:


df["origin"].unique()


# It would not make sense to perform linear regression using the "origin" column, because it contains strings.  Even if we convert those three strings to for example `0`, `1`, `2`, that is still not a good idea, because it is forcing an order on the strings, as well as forcing the spacing between them.  Instead we will add a column to the DataFrame for each possible value in this column.

# In[1]:


from sklearn.preprocessing import OneHotEncoder


# This syntax is similar to the `StandardScaler` syntax.

# In[19]:


encoder = OneHotEncoder()


# Here is our common error, of passing a one-dimensional object (a Series) instead of a two-dimensional object (a DataFrame).

# In[20]:


encoder.fit(df["origin"])


# Here is the correct way to pass this single column.

# In[21]:


encoder.fit(df[["origin"]])


# We now transform the data.  (Notice how we use the method `transform` instead of `predict`.  We are not in the Machine Learning portion at the moment, we are not predicting anything.  Instead we are in the preprocessing or the data cleaning stage.)

# In[22]:


encoder.transform(df[["origin"]])


# The produced object is a little strange, but we can convert it to a NumPy array.  (The array will contain mostly zeros, and so it can be more efficient to not store all of those rows separately.  That is why, by default, scikit-learn uses this sparse matrix object instead of a usual NumPy array.)

# In[23]:


encoder.transform(df[["origin"]]).toarray()


# There are three columns, corresponding to three values in the "origin" column.  Which column corresponds to which value?  We can use the following method to check.

# In[24]:


encoder.get_feature_names_out()


# Let's add these three new columns to our DataFrame.  To be safe, we will make the changes in a new DataFrame.

# In[25]:


df2 = df.copy()


# At some point (I don't remember exactly where), we will get an error if we have the new column names in a NumPy array instead of a list, so here we convert it to a list.

# In[26]:


new_cols = list(encoder.get_feature_names_out())


# In[27]:


new_cols


# Here we put the above NumPy array into these three new columns.

# In[28]:


df2[new_cols] = encoder.transform(df[["origin"]]).toarray()


# Think of these last three columns as like Boolean Series.  For example, everywhere we see a `1` in the "origin_europe" column, that signifies that the origin value is "europe".  These columns don't contain any new information (it was all already present in the "origin" column), but they are numeric, so they can be used for linear regression (or clustering or many other Machine Learning techniques that require numerical values).

# In[30]:


df2.sample(5, random_state=12)


# I think that on Wednesday we'll perform linear regression to predict the miles-per-gallon of a car using these new columns as well as our `cols` list of columns from before.
