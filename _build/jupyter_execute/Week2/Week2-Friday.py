#!/usr/bin/env python
# coding: utf-8

# # Week 2 Friday
# 
# Plan for today.
# * 20-30 minutes: Lecture
# * Remaining time: work on Worksheets 3-4.  Due Tuesday before Discussion Section.  Chupeng and I are here to help.
# * Reminder: Videos and video quizzes due.  Try to finish them after class if you haven't already.

# In[3]:


import pandas as pd


# In[4]:


df = pd.read_csv("indexData.csv")
df_info = pd.read_csv("indexInfo.csv")


# ## Practice with `isna`, `any`, and list comprehension

# * Make a Boolean Series indicating which columns in the indexData.csv file contain missing data.  Use `isna` and `any` with an appropriate `axis` keyword argument,   Name that Boolean Series `bool_ser`.
# 
# The best way to answer these sorts of questions is to build up the solution one step at a time.  Here is the original DataFrame.

# In[5]:


df


# Here is the result of calling the `isna` method.  We see `False` because there weren't any missing values in the portion of the DataFrame we could see.

# In[6]:


df.isna()


# We now go from the pandas DataFrame we see above to a pandas Series.  The `axis=0` says that we should plug in one full column at a time.  (We may talk later about a more general way to understand this `axis` keyword argument.  For now, just memorize that `axis=0` means to plug in a full column at a time.)

# In[7]:


df.isna().any(axis=0)


# If this `any` method is difficult to understand, maybe it will be more clear on a smaller DataFrame.

# In[9]:


df_small = pd.DataFrame([[False, True], [False, False]], columns=["A","B"])
df_small


# There was a question about data types of columns.  That could mean a few different things.  The entries in the columns are of type `bool`.

# In[10]:


df_small.dtypes


# In[11]:


df_small.columns


# The names of the columns are of type `str`, for string.

# In[12]:


[type(col) for col in df_small.columns]


# The full column is a pandas Series.

# In[13]:


type(df_small["A"])


# Back to the original topic, which was about how this `any` method works.

# In[14]:


df_small


# Here we go through each column, one at a time, and check if it contains any `True` values.  Notice how we don't use `isna` in this case.  The `isna` was to get Boolean values, but here we already have Boolean values.

# In[15]:


df_small.any(axis=0)


# Equally useful to the `any` method is the `all` method.  In this case, we get `False` for the values, because no column is entirely `True`.

# In[16]:


df_small.all(axis=0)


# If we use `axis=1`, then we are plugging in a full row at a time.  (Notice that the rows are labeled as `0` and `1`, whereas we specified when we created `df_small` that the columns should labeled as `"A"` and `"B"`.

# In[17]:


df_small.any(axis=1)


# In[18]:


df_small


# * Using list comprehension and `bool_ser.index`, make a list of all the column names which contain missing data.

# Here is a construction of `bool_ser`.  This is a pandas Series whose index contains the names of the columns and whose values are `True` or `False`, depending on whether that column had missing data or not.

# In[19]:


bool_ser = df.isna().any(axis=0)
bool_ser


# Here is an example of list comprehension.  This just creates a list with every column name.

# In[20]:


[c for c in bool_ser.index]


# We only want the names of the columns which have missing data, so we add an `if` condition.  The following is correct but it is not good style, because `boolser[c] == True` is the exact same as `boolser[c]` itself.

# In[21]:


[c for c in bool_ser.index if bool_ser[c] == True]


# Here is the more elegant version.

# In[22]:


# Better style
[c for c in bool_ser.index if bool_ser[c]]


# * Find the same columns using Boolean indexing.  (If you really want it to be a list, wrap the answer in the `list` constructor.)

# Here is a reminder of what `bool_ser` looked like.  This is a good candidate for Boolean indexing, because it is a Boolean Series.

# In[23]:


bool_ser


# Here we use Boolean indexing to keep only the key/value pairs for which the value is `True`.

# In[25]:


# Boolean indexing, both the keys and the values
bool_ser[bool_ser]


# If we only care about the left-hand terms (the index, which I also call the keys), we can use the `index` attribute.

# In[26]:


# Boolean indexing then index (only keep the keys, only keep the index)
bool_ser[bool_ser].index


# If that's confusing, maybe this example will be more clear, because the values look more interesting.

# In[29]:


test_series = pd.Series({"a": 3, "c": 10, "g": -5})
test_series


# Notice how the values (3, 10, -5) disappear when we ask for the index.

# In[30]:


test_series.index


# I wasn't totally sure this would work because I don't need to remove the keys very often, but we can get the values by accessing the `values` attribute.  (I also thought it might need parentheses like a method, but it doesn't.)

# In[31]:


test_series.values


# Notice how `bool_ser[bool_ser].index` was not a list.  If you really want it to be a list (there isn't much advantage to making it a list), then you can wrap it in the `list` constructor function.

# In[27]:


# If you insist on it being a list
list(bool_ser[bool_ser].index)


# * Make the same list using a list comprehension of the form 
# ```
# [c for c in df.columns if ???]
# ```

# Here we don't need to put an `axis` term in the `any` method, because `df[c].isna()` is a pandas Series, so it only has one dimension, so we don't need to specify a dimension when we call `any`.  (There are definitely other ways to do this.  Probably `df.isna().any(axis=0)[c]` would also work.)

# In[28]:


[c for c in df.columns if df[c].isna().any()]


# ## Boolean indexing vs `groupby`

# Here is the other DataFrame we loaded.

# In[32]:


df_info


# * Using a for loop, Boolean indexing, and an f-string (but no `groupby`), for each currency in `df_info`, print how many rows in the dataset use that currency.

# Notice how there are some repetitions.

# In[33]:


for cur in df_info["Currency"]:
    print(cur)


# We can use `df_info["Currency"].unique()` if we don't want repeated currencies.

# In[34]:


for cur in df_info["Currency"].unique():
    print(cur)


# The following is similar to what you are asked to do in Worksheet 3.  (In Worksheet 3, you are asked to provide more information than just the number of rows.)

# In[38]:


# Hint for the end of Worksheet 3
for cur in df_info["Currency"].unique():
    df_sub = df_info[df_info["Currency"] == cur]
    print(f"The currency is {cur} and the number of rows is {len(df_sub)}")


# * Print the same information, again using a for loop, this time using a pandas GroupBy object and its `count` method.

# The object `df_info.groupby("Currency")` by itself is not very useful.

# In[40]:


df_info.groupby("Currency")


#  We need to do something with this GroupBy object, like iterate over it using a for loop, or like in the following example, where we call the `count` method.  This tells us how many rows there are.  (At least in this case.  Maybe it subtracts missing values...)

# In[41]:


df_info.groupby("Currency").count()


# Notice how `df_info.groupby("Currency").count()` is a pandas DataFrame.  So we can get the "Region" column from that DataFrame just like usual, by using square brackets.

# In[43]:


gb_ser = df_info.groupby("Currency").count()["Region"]
gb_ser


# This `gb_ser` pandas Series has all the information we want.

# In[44]:


for cur in gb_ser.index:
    print(f"The currency is {cur} and the number of rows is {gb_ser[cur]}")


# * Print the same information, this time iterating over the GroupBy object as follows.
# 
# ```
# for cur, df_sub in ???:
#     print(???)
# ```

# We didn't get here!
