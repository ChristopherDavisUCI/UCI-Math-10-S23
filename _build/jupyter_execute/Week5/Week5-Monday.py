#!/usr/bin/env python
# coding: utf-8

# # Week 5 Monday

# ## Announcements
# 
# * Video quizzes have been turned into "Practice Quizzes" (same links as before on Canvas).  Scores on these practice quizzes won't affect your grade, and you can take them as many times as you want.  (About half the quizzes have multiple versions.)
# * No videos or video quizzes this Friday.  After the midterm, we'll start the **Machine Learning** portion of Math 10.
# * Midterm Wednesday.  Covers material up to and including Week 4 Friday's lecture and videos.
# * Pick up a notecard if you haven't already.  Hand-written notes, written on both sides, can be used during the midterm.
# * Sample midterm posted on the "Week 5" page on Canvas.  I'll ask William to go over as much of the sample midterm as possible on Tuesday.  Solutions posted Tuesday afternoon at the latest.  (If you want a hint on a specific question before then, please ask on Ed Discussion or come to my office hours.)
# * I have office hours after class, 11am, next door in ALP 3610.

# ## `map` vs `applymap` vs `apply`
# 
# Here are some important facts about these methods.
# 
# * `map` is a Series method.  `applymap` and `apply` are DataFrame methods.  (Secretly `apply` can also be used on a Series, but we haven't covered that.)
# * All three take a function as an input.
# * `map` and `applymap` apply that function on every entry.  `apply` applies the function on an entire row (`axis=1`) or on an entire column (`axis=0`).
# * I think everyone would agree `apply` is the most difficult to understand of the three.

# In[2]:


import pandas as pd
import altair as alt
import seaborn as sns


# In[3]:


df = sns.load_dataset("penguins")


# The following wants to multiply every element in `df` by `2`.  It doesn't work because `2*x` is not a function in Python.

# In[5]:


df.applymap(2*x)


# Here we use a lambda function to define the function $x \leadsto 2x$.  Notice how this works even for string values.

# In[6]:


df.applymap(lambda x: 2*x)


# Here is an example of getting a sub-DataFrame using `iloc`.  Notice how the above values are not reflected.  That is because we did not save the values.  (A hint is that something got displayed to the screen.)

# In[7]:


df.iloc[:2, 1:4]


# Sometimes the `inplace` keyword argument can be used, but not for `applymap`.

# In[8]:


df.applymap(lambda x: 2*x, inplace=True)


# Here is a reminder of the `rename` method.  Notice how we are also specifying `axis=0`, which tells pandas that it is the row labels which should be changed.

# In[9]:


df.rename({2:"New row name"}, axis=0)


# Again, there was no change to `df` itself.  (The row `2` is still called `2`.)

# In[10]:


df[:4]


# Here we use the `inplace` keyword argument.  Notice how nothing gets displayed.

# In[11]:


# nothing got displayed; df changed
df.rename({2:"New row name"}, axis=0, inplace=True)


# Now the row name did change.

# In[12]:


df[:4]


# Now we switch to talking about `apply`.  The use of the `axis` keyword argument is definitely confusing.  See the Week 4 videos for the most consistent description I know for when to use `axis=0` and when to use `axis=1`.

# In[13]:


df.shape


# Here we are keeping the column names the same.  The number `344` represents the length of each column.

# In[14]:


df.apply(len, axis=0)


# The number `7` here represents the length of each row.

# In[15]:


df.apply(len, axis=1)


# Here I'll try to convince you that the row names did not change.

# In[16]:


df.apply(len, axis=1).index


# In[17]:


df.index


# ### Example of `apply`
# 
# * Using `apply`, subtract a suitable constant from all the numeric columns  in the Penguins DataFrame so each numeric column has mean 0.  (Use a different constant for each column.) Use `pandas.api.types.is_numeric_dtype` to determine if a column is numeric or not.

# In[18]:


# step 1: find the numeric columns
pd.api.types.is_numeric_dtype(df["species"])


# In[19]:


pd.api.types.is_numeric_dtype(df["bill_length_mm"])


# It is cumbersome to have to write `pd.api.types.is_numeric_dtype` each time, so we can import it separately.

# In[20]:


from pandas.api.types import is_numeric_dtype


# In[21]:


is_numeric_dtype(df["bill_length_mm"])


# In[22]:


is_numeric_dtype(float)


# Here is a first step of list comprehension, where we just get the list version of `df.columns` (we are not removing anything).

# In[24]:


# Using list comprehension, get a list of all the numeric columns in df
[c for c in df.columns]


# Now we add an `if` condition to just keep the numeric columns.  Notice how above we typed `is_numeric_dtype(df["bill_length_mm"])`, and `c` is some value like `"bill_length_mm"`, so we should use `df[c]`.

# In[25]:


# Using list comprehension, get a list of all the numeric columns in df
[c for c in df.columns if is_numeric_dtype(df[c]) == True]


# The `== True` is not doing anything here, so we can remove it.

# In[27]:


num_cols = [c for c in df.columns if is_numeric_dtype(df[c])]
num_cols


# There was a question about whether `c` should be in quotation marks.  It is an important distinction.  If we put `c` in quotation marks, `"c"`, then we just get the letter "c", and the value of the variable `c` is irrelevant.

# In[28]:


for c in df.columns:
    print("c")


# Here is the same code without quotation marks around `c`.

# In[29]:


for c in df.columns:
    print(c)


# Here we get the sub-DataFrame consisting of the numeric columns, using `df.loc`.

# In[30]:


df.loc[:, num_cols]


# The following is an abbreviation for the same thing.

# In[31]:


df[num_cols]


# Let's save this sub-DataFrame.

# In[32]:


df_sub = df[num_cols]


# In[33]:


df_sub


# We want to use some code like the following.
# 
# ```
# # Subtract a constant from each column so the mean is 0
# df_sub.apply(???, axis=???)
# ```

# Here is the correct formula.

# In[35]:


# We're plugging in a column at a time, so use axis=0
df2 = df_sub.apply(lambda x: x-x.mean(), axis=0)


# The code `x-x.mean()` can be confusing, because `x` represents a pandas Series and `x.mean()` represents a number.  Here is a specific example of computing pandas Series minus a number in Python.

# In[37]:


df["bill_length_mm"] - 30


# We want the columns to have mean 0.  The following says that (up to numerical precision issues), these means are 0 (or at least very close to 0).

# In[36]:


df2.mean(axis=0)


# ## More examples

# This doesn't always work (for example, some websites block it), but sometimes pandas can read tables directly from websites.  The `pd.read_html` function returns a list of html tables (as DataFrames) that pandas can find on the website.
# 
# Here are some websites for which `pd.read_html` does work.
# 
# * [https://en.wikipedia.org/wiki/Irvine,_California](https://en.wikipedia.org/wiki/Irvine,_California)
# * [https://www.usclimatedata.com/climate/irvine/california/united-states/usca2494](https://www.usclimatedata.com/climate/irvine/california/united-states/usca2494)
# * [https://www.usclimatedata.com/climate/new-york/new-york/united-states/usny0996](https://www.usclimatedata.com/climate/irvine/california/united-states/usca2494)

# In[38]:


link = "https://www.usclimatedata.com/climate/irvine/california/united-states/usca2494"


# The function `pd.read_html` returns a list of DataFrames.

# In[39]:


df_list = pd.read_html(link)


# In[40]:


df_list[0]


# In[41]:


df_list[1]


# For the above two DataFrames, would it make more sense to stack them side-by-side or on top of each other?  Definitely side-by-side, because we want the months to go from January to December.  So we are changing the columns axis (the number of columns is going from 7 to 14), so we specify `axis=1`.

# In[42]:


pd.concat((df_list[0], df_list[1]), axis=1)

