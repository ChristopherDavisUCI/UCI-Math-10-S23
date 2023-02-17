#!/usr/bin/env python
# coding: utf-8

# # Week 2 Monday

# ## Announcements
# 
# * In-class Quiz 1 is in discussion tomorrow.  Based on the material up to and including Worksheet 2.  (So the material from last week's videos will not be covered, at least not directly.)
# * Worksheet 1 and Worksheet 2 due before discussion tomorrow.
# * I have office hours after class, 11am, next door in ALP 3610.
# * Friday videos and video quizzes for this week are posted.

# The main goal today is to briefly introduce two topics that appear on Worksheet 3: f-strings and `map`.  Also we will briefly see how to time a computation.

# In[3]:


import pandas as pd


# ## f-strings
# 
# f-strings are a relatively recent addition to Python.  We are working in Python 3.9 (how do I know?) and they were added in Python 3.6, which was released in 2016.  For this reason, you will often see Python code (even code written by experts) that uses the previous `format` method, so it's important to recognize both.
# 
# * Define `name` to be `"Chris"` and `day` to be today's date as a pandas Timestamp object.

# In[4]:


name = "Chris"
day = pd.to_datetime("10/3/2022")


# Let's check that `day` really is a Timestamp and not a string.

# In[5]:


day


# In[6]:


type(day)


# * Print `"Hello, Chris, how are you doing on Monday?"` using `name` and `day` and the string method `format`.

# Even though the f-string approach is more elegant and more readable than the `format` approach, it is still useful to be able to recognize the `format` approach.  Code from before 2016 (and even a lot of modern Python code) will use the `format` approach.

# In[8]:


# old way
s = "Hello, {}, how are you doing on {}?".format(name, day)
print(s)


# The above showed the Timestamp, not the day of the week.  Here we fix that.  Notice how we don't need the `dt` accessor, because we are not working with a pandas Series.  We only use `.dt` and `.str` when working with a pandas Series.

# In[9]:


# don't need `dt` because `day` is not a Series
s = "Hello, {}, how are you doing on {}?".format(name, day.day_name())
print(s)


# * Print `"Hello, Chris, how are you doing today on October 3?"` using `name` and `day` and f-strings.  (You can also have the f-string make the conversion to "Monday" automatically, using a [strftime format code](https://strftime.org/).)

# The biggest drawback to the `format` code is that you can't read the text in order (you have to jump to the right to find what goes inside the brackets `{}`).  Also, when there are lots of variables, you need to count them and that can get confusing.
# 
# The f-string approach is much more readable.  Notice how we add the letter `f` before the quotation marks.

# In[10]:


# new way, f-string way
s1 = f"Hello, {name}, how are you doing on {day.day_name()}?"
print(s1)


# The type of `s1` in Python is still just an ordinary string.

# In[11]:


type(s1)


# The following is not nearly as important for Math 10 as the `day_name` method, but just for fun, here is a way to use a format code instead of `day_name` to display the day of the week.  (See the link above for other options.)

# In[12]:


# just for fun, day_name is more important
f"Hello, {name}, how are you doing on {day:%A}?"


# ## Adding a column to a pandas DataFrame
# 
# Here we briefly see two things.
# * How to insert a new column into a DataFrame.
# * How to use `map` to apply a function to every value in a pandas Series.

# The string `split` method converts from a string to a list.  If you don't pass any arguments to the method, meaning you call `split()` with empty parentheses, then it will divide the string at all whitespace.

# In[13]:


df = pd.DataFrame(
    {
        "A": [3,1,4,1,5],
        "B": "Hello, how are you doing?".split()
    }
)

df


# Making a new column with the same value in every position is easy.

# * Make a new column "C" that contains `4.5` in every position.

# In[14]:


df["C"] = 4.5


# Here we verify that there really is a new column in `df`.

# In[15]:


df


# Remember that if we use the notation `df[???]`, without `loc` or `iloc`, that will access a column by its label.

# In[16]:


df["B"]


# Using an integer position here doesn't work.  (This would work if there were a column with label `1`.)

# In[17]:


# looking in columns
df[1]


# If instead you want to get a row, not a column, you should use `loc`.

# In[18]:


# looking in rows
df.loc["B"]


# I don't use this approach very often, but here is the `loc` way to get a column.

# In[19]:


df.loc[:,"B"]


# We can also create a new column using this approach.  (I wasn't 100% sure this would work, since I always use the abbreviation without `loc`, but it did work.)

# In[20]:


df.loc[:, "New Column"] = 3


# Now we can see that two new columns have been added to `df`.

# In[21]:


df


# * Using the pandas Series `map` method and a lambda function, make a new column "D" that contains the first two characters of each string in column "B".

# Here is a reminder of what `df["B"]` looks like.  Each value is a string in this pandas Series.

# In[22]:


df["B"]


# The `map` method is a common place where we use lambda functions.  (Any function will work; it doesn't have to be a lambda function.)  The input to `map` is the function we want to apply to every value.  Here we use slicing to get the first two characters of every string in the "B" column.

# In[23]:


df["B"].map(lambda s: s[:2])


# Here we assign that pandas Series to a new column.

# In[24]:


df["D"] = df["B"].map(lambda s: s[:2])


# In[25]:


df


# * Make a new column "E" that is equal to column "A" divided by pi.  (Get pi from the NumPy library.)  Try this using `map` and also by dividing the column directly.  The direct method (without `map`) is more efficient (at least for large Series) and more readable.

# Unlike Matlab and Mathematica, Python does not have a built-in constant `pi`.  (In general, Python has much less built-in math functionality.)

# In[26]:


pi


# We will get the definition of `pi` from the NumPy library.  The NumPy library is very important, and is in the background of many efficient pandas computations.

# In[27]:


import numpy as np


# In[28]:


np.pi


# Don't overuse `map`.  Even though the following works, there is a much simpler approach.

# In[29]:


df["E"] = df["A"].map(lambda x: x/np.pi)


# In[30]:


df


# Here is the better approach.  We just divide the column by `np.pi`, and pandas automatically "broadcasts" the operation to each entry in the column.

# In[31]:


# better way
df["E"] = df["A"]/np.pi


# In[32]:


df


# As a quick aside, if you want to time a computation, one way is to put `%%time` at the top of the cell.  With a much larger column (say with ten million numbers in it), we would find that this broadcasting approach is faster.

# In[33]:


get_ipython().run_cell_magic('time', '', 'df["E"] = df["A"]/np.pi\n')


# * Check that operating on the column directly really is more efficient for longer Series.  Make a pandas Series `ser` whose values are the even numbers from `0` up to (but not including) `10**7`.  Put `%%time` in the top of the cell (this is called a Jupyter or IPython "magic" function) and time how long it takes to divide each value in `ser` by pi.  Try with both methods from above.

# We didn't get here!
