#!/usr/bin/env python
# coding: utf-8

# # Week 0 Friday
# 
# Starting in Week 1, we will have more specific topics.  Today, we went through a variety of basic topics, especially related to the *type* of an object in Python.

# We start by importing pandas.  The pandas library is the most important library for Math 10.  (The second-most important library is probably scikit-learn, which we will use extensively in the Machine Learning portion of Math 10.)
# 
# In theory, you could give pandas an abbreviation other than `pd`, or not use any abbreviation at all, but in practice, everyone uses `pd`, and we will also always use `pd`.

# In[3]:


import pandas as pd


# Just to emphasize that `pd` is now defined but `pandas` is not, because of our import statement.

# In[4]:


pandas


# I use the terms "module" and "library" moreorless interchangeably.  I usually refer to pandas as a "library"; here Python is referring to it as a "module".

# In[5]:


pd


# One of the most important concepts at the beginning of Math 10 is the concept of the `type` of an object in Python.  Different types of objects have different functionality associated with them.  To use the `read_csv` function defined by pandas, we need to use as an argument an element with the type of string.  Here we forget to use quotation marks (i.e., we forgot to turn vend.csv into a string), so that's why we get an error.

# In[6]:


pd.read_csv(vend.csv)


# Instead of using `vend.csv` as our argument, we use `"vend.csv"`.  This `"vend.csv"` is a string.

# In[7]:


pd.read_csv("vend.csv")


# To later access the contents of this dataset, we should store it in some variable name.  A good default choice is `df`.

# In[8]:


df = pd.read_csv("vend.csv")


# In the worksheet from yesterday, we only did a few things with this dataset.  One thing we did was to look at its first 10 rows using the `head` method.

# In[9]:


df.head(10)


# Here is an example of how different data types have different functionality.  We can check an elements data type by using the built-in Python function `type`.  (Python has relatively few built-in functions, definitely many fewer than Mathematica, for example.  Often the functions we use in Math 10 will come from an external library.  For example, the `pd.read_csv` function is a function defined in the pandas library.)

# In[10]:


name = "Chris"


# In[11]:


type(name)


# The data type of `df` is a type defined in `pandas`.  The following says `pandas.core.frame.DataFrame`.  I usually ignore the middle terms and only focus on the first and last term.  The first term is telling us that this is defined in pandas, and the last term is telling us that the `type` of `df` is `DataFrame`.

# In[12]:


type(df)


# As mentioned above, different types of objects have different functionality.  The following are examples of all the attributes and methods of strings.  (Technically methods are themselves a type of attribute.)  We'll see examples of how to use these attributes and methods soon.  I recommend mostly ignoring the ones that begin with two underscores, like `__add__`; those are mostly just used in the background by Python.

# In[13]:


dir(name)


# In retrospect, the following was a bad example, because `"Chris"` was already capitalized.

# In[14]:


# using the capitalize method
name.capitalize()


# In[15]:


name


# Here is a better example.  We call the `upper` method of `name` to convert the string to all upper-case letters.

# In[16]:


name.upper()


# If we try to call the `upper` method on `df`, we get an error, because DataFrames do not have an `upper` method.  This is an example of how knowing the data type that you're working with is important, because once you know what the data type is, you also know the special functionality you can access.

# In[17]:


df.upper()


# Here is an example of an *attribute*, as opposed to a method.  Methods are like functions, and attributes are like variables.  That's not a perfect description; you will get more familiar with attributes and methods, and get more intuition about whether something is a method or attribute, as you work more in Python.  This particular attribute records the number of rows and the number of columns of `df`.

# In[18]:


# attribute, not a method
df.shape


# The value corresponding to `df.shape` is what is called a *tuple*.  At first glance, tuples are very similar to lists (which showed up in Worksheet 0).  We will discuss some differences later.

# In[19]:


type(df.shape)


# Here is another very useful attribute of a DataFrame, the `columns` attribute.  This tells us all the different column names.  (I'm being careful to not say it's a list of the column names, because it is not a list, nor is it a tuple...)

# In[20]:


df.columns


# To access a specific column from a pandas DataFrame, we can use the following syntax.

# In[21]:


df["RPrice"]


# Columns in pandas DataFrames are represented by the type Series.  The two most important data types in pandas are DataFrames and Series.

# In[22]:


type(df["RPrice"])


# Indexing in Python starts at 0, so if we want to get the first element of something, like a pandas Series, I will usually call it the "zeroth" element instead of the "first" element.  This syntax looks a little strange at first, but to get the zeroth element in the pandas Series `df["RPrice"]`, we use `.iloc[0]`.

# In[23]:


df["RPrice"].iloc[0]


# The type of this value is given as `numpy.float64`.  NumPy is another Python library (like pandas), and because NumPy is a dependency of pandas, any system where pandas works should also have NumPy installed.  This data type is defined in NumPy.  The "float" in `numpy.float64` is telling us that these are decimals (as opposed to integers).  The "64" is specifying how much space the numbers take up; the "64" won't be very important for us in Math 10.
# 
# The following code is longer than what we were writing above.  If it doesn't make sense, try to break it up into separate pieces.  In this case, to understand the full code, you should first understand `df["RPrice"]`, then you should understand `df["RPrice"].iloc[0]`, and once you understand that, the full code should make sense, `type(df["RPrice"].iloc[0])`.

# In[24]:


type(df["RPrice"].iloc[0])


# The list of attributes and methods of a pandas Series is quite a bit longer than the list for strings that we saw above.

# In[25]:


dir(df["RPrice"])


# Notice that there is a `dtype` attribute.  That gives us easier access to the data type of the contents of the pandas Series.

# In[26]:


df["RPrice"].dtype


# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=05525cbf-df08-4927-98e9-518b8fa61854' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>
