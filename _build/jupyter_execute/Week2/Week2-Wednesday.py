#!/usr/bin/env python
# coding: utf-8

# # Week 2 Wednesday

# ## Announcements
# 
# * Change to Wednesday Office Hours Location: Let's try ALP 2800 Wednesday (today) 1pm.
# * Worksheet 4 uses some material from Worksheet 3, so definitely start with Worksheet 3.
# * Reminder: Week 2 videos and video quizzes are due before lecture on Friday.
# * On Friday I hope to introduce plotting in Python, hopefully briefly introducing each of Matplotlib (similar to plotting in Matlab), Altair, Seaborn, and Plotly (Altair, Seaborn, and Plotly are similar to each other).

# In[3]:


import pandas as pd


# In[4]:


df = pd.read_csv("indexData.csv")


# ## Review with Boolean indexing and `map`
# 
# * Define `df_sub` to be the sub-DataFrame of `df` which corresponds to the Hong Kong stock exchange (abbrev: "HSI") in 2008 (last two characters in the "Date" column "08").
# 
# **Comment**.  It would be more robust to convert the Date column to datetime type, like you need to do in Worksheet 3 and in Worksheet 4, but what we're doing here is easier. **Do not** use `pd.to_datetime(df["Date"])` in this particular example, because all of the dates from before 2000 will be incorrect.

# Let's remind ourselves which columns are in this dataset.

# In[5]:


df.columns


# Don't try to solve a question like this one all at once.  Instead, break it into parts.  Here we make a first Boolean Series.

# In[11]:


# Boolean Series for Hong Kong stock exchange
ser1 = (df["Name"] == "HSI")
ser1


# Here we get ready to make the second Boolean Series.

# In[8]:


df["Date"].map(lambda s: s[-2:])


# Why do we get `False` at the top, since four of these rows are equal to 66?  The problem is that the Series contains strings, not integers.  (Notice how the `dtype` is reported as `object`, which in this case is a hint that the elements are strings.)

# In[9]:


df["Date"].map(lambda s: s[-2:]) == 66


# Here we are careful to use a string instead of an integer when we check for equality.  We also switch to `"08"`, since that is the year we were interested in.

# In[13]:


ser2 = (df["Date"].map(lambda s: s[-2:]) == "08")
ser2


# Now we are finally ready to use Boolean indexing.  Remember that you can combine two Boolean Series elementwise using `&`.

# In[14]:


# Boolean indexing
df_sub = df[ser1 & ser2]
df_sub


# ## Using `copy` to prevent a specific warning

# * Create a new column in `df_sub`, named "Month", corresponding to the month, as a string like "July".  What warning shows up?  How can we avoid this warning in the future?

# In[16]:


df_sub["Month"] = pd.to_datetime(df_sub["Date"]).dt.month_name()


# Briefly, this warning is showing up because `df_sub` might not be its own DataFrame, but instead might be a "view" of a portion of the original `df` DataFrame.
# 
# Don't worry about that.  Instead know that we can avoid this warning by returning to the line where we created `df_sub` and using the `copy` method.

# In[17]:


df_sub = df[ser1 & ser2].copy()


# Now the exact same call as above works without raising a warning.

# In[18]:


df_sub["Month"] = pd.to_datetime(df_sub["Date"]).dt.month_name()


# Let's check that we really do have a new "Month" column.

# In[19]:


df_sub.head(3)


# ## Finding missing values with `isna`

# * Is there any missing data in this sub-DataFrame?  In how many rows?  In what rows?

# Missing values in pandas and in NumPy are denoted by some variant of `np.nan`, which stands for "not a number".  We can detect them by using the `isna` method.

# In[20]:


df_sub.isna()


# If all we care about is, are there *any* missing values in a row, we can use the `axis=1` keyword argument to the `any` method.  We may talk about `axis=1` more later, but for now, just know that it means we are plugging in one row at a time (as opposed to one column at a time).

# In[21]:


# Which rows have missing data?
df_sub.isna().any(axis=1)


# Here we count how many rows in `df_sub` have missing data.

# In[22]:


# How many rows have missing data?
df_sub.isna().any(axis=1).sum()


# Here we use Boolean indexing to keep only those rows with missing data.

# In[23]:


# Use Boolean indexing to keep the rows with missing data
df_sub[df_sub.isna().any(axis=1)]


# Back to the `axis` keyword argument.  Here is what happens if we use `axis=0`, so we are plugging in one column at a time.

# In[24]:


# axis=0 Which columns have missing data?
df_sub.isna().any(axis=0)


# * Aside: How can you take the element-wise negation of a Boolean Series?  For example, which columns *do not* have missing values?

# To take an elementwise negation in NumPy or pandas, use the tilde symbol, `~`.

# In[25]:


~df_sub.isna().any(axis=0)


# ## The DataFrame method `groupby`
# 
# When I first learned the `groupby` method, I found it very confusing.  I wanted to know, is `df_sub.groupby("Month")` a DataFrame?  A list of DataFrames?  But this isn't really the correct *object-oriented* perspective.  In fact, `df_sub.groupby("Month")` is just its own special type of object, a pandas GroupBy object.
# 
# There are two things I want you to know about GroupBy objects.
# 
# 1.  If you apply a method like `mean`, what is the result?
# 2.  If you iterate over a GroupBy object, what values do you get?

# * Call `df_sub.groupby("Month").mean()`.  What information is this conveying?

# For example, in the following, the `24608` at the top left means that for the month of April, the average value in the "Open" column was approximately 24608.

# In[26]:


df_sub.groupby("Month").mean()


# * What if you don't want the months sorted alphabetically?
# 
# It's often helpful that the values get sorted, but in this case, we don't want them sorted alphabetically.  By using the `sort=False` keyword argument, we tell pandas to keep whatever the original DataFrame order was.

# In[27]:


df_sub.groupby("Month", sort=False).mean()


# * Call the following code, replacing `???` with appropriate values.  What information is this conveying?
# 
# ```
# for gp, df_mini in df_sub.groupby("Month"):
#     print(???"The month is {gp} and the number of rows is ???.")
# ```

# We just raced through this example, but I hope it gives a hint of what happens when you iterate over a pandas GroupBy object.  The value I call `df_mini` is the sub-DataFrame corresponding to a particular value (a particular value of "Month" in this case).  For example, because February is the shortest month, there are the fewest rows for the February mini DataFrame.

# In[28]:


for gp, df_mini in df_sub.groupby("Month"):
    print(f"The month is {gp} and the number of rows is {df_mini.shape[0]}.")

