#!/usr/bin/env python
# coding: utf-8

# # Week 1 Monday

# ## Announcements
# 
# * My office hours this quarter will be Monday 11am ALP 3610 (next door), Wednesday 1pm ALP 3600, and by appointment.  I can't reserve these computer labs for office hours; I'll send an email to announce any last-minute changes to the location.
# * Worksheet 0 is due Tuesday before discussion section.  (Need groupmates?  Today's class is a good time to find them!)
# * Videos and video quizzes for this week have been posted.  They are due before lecture on Friday.  The video quizzes are open book, open computer, and can be taken two times.  It's fine to discuss them with classmates.
# * Today will be about half lecture and half time to work on Worksheet 1.  One of our two Learning Assistants, Alicia, is here to help during the work time.
# * Usually Tuesday discussion will have an in-class quiz, but not this week.  Some new material will be introduced in Tuesday discussion this week, so be sure to attend!

# In[3]:


import pandas as pd


# In[4]:


df = pd.read_csv("vend.csv")


# ## Warm-up: Two useful pandas Series methods
# 
# Reminder: If we want to get a column named `"Product"` out of a pandas DataFrame, we can use the following syntax.  The result is a pandas Series.

# In[5]:


df["Product"]


# Here is the first of the two pandas Series methods we want to introduce.  (Think of a method like a function, but it's a function that is attached to a specific object.  In this case, the `value_counts` method is attached to our pandas Series.)  This `value_counts` method tells us how often each value occurs.

# In[6]:


df["Product"].value_counts()


# We can also tell how many items appear in the "Product" column, 170, by looking at the length of the above Series.  If all you care about is what items occur, and you don't care about how often they occur, then it makes sense to use a different pandas Series method, `unique`.

# In[7]:


df["Product"].unique()


# Here we check how many items occur, by calling the `len` function on `df["Product"].unique()`.  Notice how `len` goes at the front of this expression; that's why we're calling `len` a function rather than a method.

# In[8]:


len(df["Product"].unique())


# Notice how this answer, 171, is one more than our length of 170 for the `value_counts` output.  This is presumably because the `unique` method includes the possibility of a missing value, and the `value_counts` method does not.  (A student later pointed out to me that the `unique` output does include `nan`, which stands for "not a number", and represents missing values.)

# ## Indexing for pandas Series
# 
# Indexing for pandas Series and especially for pandas DataFrames takes some getting used to.  What various types of indexing represents is largely something that needs to be memorized.
# 
# Here is a reminder of what the top three rows in our DataFrame look like.

# In[10]:


df.head(3)


# Here is the code we used last week to get the zeroth element (remember that counting in Python starts with zero) out of the "RPrice" column.

# In[9]:


# Get the zeroth element out of the RPrice column
df["RPrice"].iloc[0]


# It's very reasonable to ask why we used this strange `.iloc[0]` syntax, when some similar versions also work in this case.  For example, `.loc[0]` also works in this case.

# In[11]:


df["RPrice"].loc[0]


# In fact, even `[0]` also works in this case.

# In[12]:


df["RPrice"][0]


# The `.iloc[0]` version is really the correct version, and the other two only work by coincidence.  Let's look more closely at this pandas Series (this column from the DataFrame) `df["RPrice"]`.  Notice there are seemingly two columns of numbers.  The left-hand column shows what's called the *index* of the pandas Series, and the right-hand column (which should be thought of as the more important column) contains the *values* of the pandas Series.  The elements in the index should be thought of as labels.  For example, `0` is the label for the float `3.5` at the top.

# In[13]:


df["RPrice"]


# Let's try sorting these prices, from biggest to smallest.  We use the `sort_values` method, which by default sorts the values from smallest to biggest, so we include what's called a *keyword argument* `ascending=False` to tell pandas to instead sort them in decreasing order.

# In[17]:


s = df["RPrice"].sort_values(ascending=False)


# Notice how the index for the new Series `s` seems to be in a scrambled order now.  That is because the index has not been sorted, only the prices.

# In[18]:


s


# The `3.5` we saw earlier at the top still has the same label `0`, and if we call `s[0]`, we do not get the top element in `s`, instead we get the element (or theoretically elements) with label `0`.

# In[19]:


s[0]


# The same goes for `loc`.  Think of `loc` as indexing by label and think of `iloc` as indexing by integer position.

# In[20]:


s.loc[0]


# For this Series `s`, to get the top value, the indexing we need to use is `iloc`.

# In[21]:


s.iloc[0]


# As an aside, if you need to access the labels directly, it is stored in the `index` attribute of the pandas Series.

# In[27]:


s.index


# We'll see the exact same `loc` vs `iloc` distinction in the context of pandas DataFrames.  There also are some additional ways to index on DataFrames, related to the fact that DataFrames are two-dimensional objects, as opposed to Series which are one-dimensional objects.

# ## Indexing for pandas DataFrames
# 
# The `loc` and `iloc` indexing works basically the same for pandas DataFrames.  Look down a few cells at our `df` DataFrame.  If we go to row `2` (counting from zero) and column `4` (counting from zero), do you agree that we reach the value `'Takis - Hot Chilli Pepper & Lime'`?  (Do not include the column names when you start counting, that should be considered as a header, and do not include the index (0, 1, 2, ...) at the left-most side when counting.)

# In[23]:


df.iloc[2,4]


# If we instead want to index using the labels instead of the integer positions, then we use `loc` instead of `iloc`.  The row value stays the same since it is both the label and the integer position (that was what caused the confusion above with the `df["RPrice"]` Series), but now the column value changes from the integer `4` to the string `"Product"`.

# In[25]:


df.loc[2,"Product"]


# In[22]:


df


# At the very top of the DataFrame, we see the column names displayed.  It's often useful to have access to these column names, and they are stored in the DataFrame's `columns` attribute.  (Aside.  Notice that this `df.columns` is not a list nor a pandas Series nor any other data type we have met before.  Python is filled with many special-purpose types of objects.  This `df.columns` is a pandas Index object, but I don't think we will use any special features of this type of object in Math 10, so we will mostly ignore it.)

# In[26]:


df.columns


# At the very left-hand side of the DataFrame, the row labels are displayed.  These labels are stored in the `index` attribute.  (Aside. I expected when I evaluated this to see something like `Index([0, 1, 2, 3, ..., 6443, 6445])`, but instead we see `RangeIndex(start=0, stop=6445, step=1)`.  The distinction is not important for us; presumably the one used by pandas in this case is more memory efficient, since it does not need to store all the explicit values.)
# 
# These row labels are important, but we won't use them quite as often as we will use the column labels.

# In[28]:


df.index


# Let's see some more types of indexing for pandas DataFrames.  There is no way to know how this works in advance.  Some of these conventions seem a little contradictory to me (the fact that some types of indexing access columns and some types of indexing access rows), and it's just something you need to memorize.
# 
# If you want to access a sub-DataFrame containing only certain columns, you can pass a list of those column names.  Repetitions are fine.  Notice how we have two pairs of square brackets.  The outer pair starts the indexing, and the inner pair creates a Python list.

# In[29]:


df[["RPrice", "RPrice", "Location"]]


# Here is an example of what is called *slicing*.  Notice how this accesses rows, not columns.  The following is short-hand for "get the first 7 rows".

# In[31]:


df[:7]


# Above when we wrote `df[:7]`, that was an abbreviation for `df[0:7]`, so the following returns the exact same sub-DataFrame.  Related to the fact that numbering in Python starts at zero, when we specify endpoints in Python, usually the left endpoint is included but the right endpoint is not included.  In this case, the zeroth row is included but the seventh row (i.e., the row at integer position `7`, it would also be reasonable to call this the eighth row) is not.

# In[30]:


df[0:7]


# Let's confirm that we really are getting seven rows.  If you call the function `len` on a pandas DataFrame, you get the number of rows.  (That's different from the Matlab convention.  In Matlab, calling `length` returns the length of the longest dimension.)

# In[32]:


len(df[:7])


# Another way to find the number of rows is to first get the tuple representing the shape of the DataFrame.

# In[33]:


df[:7].shape


# And then get the zeroth element out of this tuple, which represents the number of rows.  (If instead we used `.shape[1]`, that would get the number of columns.)

# In[34]:


df[:7].shape[0]


# ## Boolean indexing
# 
# The most powerful type of indexing for us will be what is called *Boolean indexing*.  Let's see an example on a smaller DataFrame.  Here we make `df2` by hand, instead of by reading in the contents of a csv file.  This isn't the most important part of this section, but I'll just point out that we are passing a dictionary to the DataFrame constructor function, `pd.DataFrame`.  The keys in this dictionary will be the column names, and the values in this dictionary will be the values in the columns.  There are many other things we could pass to `pd.DataFrame` instead of a dictionary.

# In[35]:


df2 = pd.DataFrame({
    "A": [5,1,6,1,2,3,8],
    "B": [2,4,2,8,3,-1,0],
    "C": [5, 1.1, 3.4, 5.5, 2.1, 4.2, -3.5]
})


# Here is the resulting DataFrame, `df2`.

# In[36]:


df2


# The point of this section is not constructing DataFrames but instead the point is how to use Boolean indexing.  Before we get to Boolean indexing, here is an example of a *Boolean Series*.  It represents, in which positions is the value in column "A" strictly greater than 4?

# In[37]:


df2["A"] > 4


# Here is an example of Boolean indexing.  Inside of the square brackets is the exact Boolean Series we saw above.  We are forming a sub-DataFrame consisting of all the rows in which the Boolean Series is `True`.  In other words, we are forming the sub-DataFrame of `df2` consisting of all rows where the value in the "A" column is strictly greater than 4.
# 
# It is worth looking at this and the next example slowly, until you understand how they work.

# In[38]:


df2[df2["A"] > 4]


# We can use the same type of Boolean indexing with a Series instead of a DataFrame on the outside.  Here we are getting all the values in the "B" column corresponding to rows where the value in the "A" column is strictly greater than 4.

# In[39]:


df2["B"][df2["A"] > 4]


# Here is a reminder of how the original DataFrame looked.  Try to convince yourself that the pandas Series output by the above cell is exactly the sub-Series of column "B" corresponding to rows where the "A" column value is strictly greater than 4.

# In[40]:


df2


# Let's see a more advanced example, still using Boolean indexing.  Here is the goal.  (This will help with a problem on the corresponding worksheet.)
# 
# * Find the sub-DataFrame where the value in column "C" is strictly greater than 2 and where the value in column "A" column is not equal to 1.
# 
# Our strategy will be to make one Boolean Series for the column "C" restriction, then make another Boolean Series for the column "A" restriction, and then to combine them.

# In[41]:


df2["C"] > 2


# To check "not equal" in Python, we can use `!=`.

# In[42]:


df2["A"] != 1


# We now create the Boolean Series where the previous two Boolean Series are both `True`.  If you've done some Python programming before, you might expect to use the word `and`, and that is correct in base Python, but in pandas (and in NumPy) usually we want instead to use the ampersand symbol, `&`.

# In[43]:


# Warning: in pandas, use &, in base Python, use and
(df2["C"] > 2) & (df2["A"] != 1)


# We now use Boolean indexing with the Boolean Series we just created.  The following looks complicated, but the part inside the brackets is exactly the above code.

# In[44]:


df2[(df2["C"] > 2) & (df2["A"] != 1)]


# The above output represents the sub-DataFrame of `df2` consisting of all rows for which the "C" value is strictly greater than 2 and for which the "A" value is not equal to 1.
