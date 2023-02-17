#!/usr/bin/env python
# coding: utf-8

# # Week 3 Friday

# ## Announcements
# 
# * Videos and video quizzes due.  Finish them later if you haven't already.
# * Worksheet 3 is graded.  (If you got a 0 and submitted something, probably you did not enable public sharing.  Enable public sharing following the instructions on the bottom of Worksheet 3 and send me an email to let me know.)
# * Worksheets 5 and 6 are due Tuesday before discussion section.
# * The in-class quiz on Tuesday will be based on the material up to and including Worksheet 6

# ## Plan
# 
# The main goal of today is to see how to use `groupby` with a for loop.  (I have had that on the agenda several times but never gotten to it.)  We will have plenty of time for questions, so please ask when something doesn't make sense.
# 
# * Overall question: Within the "mpg" dataset from Seaborn, do newer cars seem to become more fuel-efficient (i.e., have higher mpg)? 

# In[1]:


import pandas as pd
import seaborn as sns


# In[2]:


df = sns.load_dataset("mpg")


# In[3]:


df.head(3)


# ## Using `groupby` with a for loop
# 
# Recall that `df.groupby` produces its own type of object.  I often abbreviate the long name to a pandas GroupBy object.

# In[4]:


x = df.groupby("model_year")
x


# Here's a brief reminder about for loops in an easier example.  Here we are iterating over a range object.  Later we will iterate over a pandas GroupBy object.

# In[6]:


for z in range(5):
    print(z)


# If we call `print` on the `type` of an object, we see an expression like the following.  As far as I know, `<class 'int'>` is exactly the same as `int`.

# In[5]:


for z in range(5):
    print(type(z))


# It would be very reasonable to think that `z` is not defined, because it only occurred as a variable inside a for loop, but that's not how Python works with respect to for loops.  The variables are accessible external to the for loop.  We only see `4` because the assignment `z=4` overwrote the earlier assignments `z=0`, then `z=1`, and so on.

# In[7]:


z


# This is what we usually see when we check the `type` of an object.

# In[8]:


type(z)


# But if we `print` the type of the object, we see the same expression we saw above.  (**Comment**.  This material about `print(type(z))` is not itself important for us.  The only reason we are covering it, is so you are not confused when we see the word `class` below.)

# In[9]:


print(type(z))


# Let's finally get to the topic of today, which is what type of objects do we get when we iterate through a pandas GroupBy object.  We get tuples.

# In[10]:


for z in df.groupby("model_year"):
    print(type(z))


# The most natural question to ask then is, what is the length of these tuples?  The length is 2.

# In[11]:


for z in df.groupby("model_year"):
    print(len(z))


# What is the initial element of the length 2 tuple?  Here they are.  Notice that these values are exactly the "model_year" values that occur in our pandas DataFrame `df`.

# In[13]:


for z in df.groupby("model_year"):
    print(z[0])


# We will usually "unpack" the tuple into two separate variables.  The `a` value in the following is exactly the same as the `z[0]` value above.

# In[12]:


for z in df.groupby("model_year"):
    (a,b) = z # tuple unpacking
    print(a)


# Here is a more elegant way to do the tuple unpacking; we unpack immediately when we create the for loop, as opposed to defining a variable `z` which we never use.

# In[14]:


for a,b in df.groupby("model_year"):
    print(a)


# This tuple unpacking only works because the elements produced when we iterate over `df.groupby` are length 2 tuples.  If we try to do this same unpacking with our `range(5)` for loop, we get an error.

# In[15]:


for a,b in range(5):
    print(a)


# We get the exact same error if we try to unpack the initial thing in that for loop, `0`.

# In[16]:


a,b = 0


# Why do we see `2` so many times?  Does the following help?  Initially `z` is assigned to be `(3,5)`, and its length is `2`.  Then `z` is assigned to be `(1,2,3,4)`, and its length is `4`, and so on.  In our GroupBy for loop, the values of `z` always have the same length.

# In[17]:


for z in [(3,5), (1,2,3,4), (1,6,5)]:
    print(len(z))


# We've already seen the following example.  The initial element (the zeroth element) in the length 2 tuples is the value we are grouping by.

# In[18]:


for a,b in df.groupby("model_year"):
    print(a)


# If we try to do the analogous thing for `b`, the second object in the tuple, it is much less helpful.  We can at least tell that `b` is much more complicated than `a`.

# In[19]:


for a,b in df.groupby("model_year"):
    print(b)


# In[20]:


for a,b in df.groupby("model_year"):
    print(type(b))


# Instead of using `print`, let's use `display` for `b`.  The information is the same, but it will be in a more familiar presentation.  Let's also use the expression `break` to leave the for loop immediately.  (So here we are doing the code inside the for loop exactly one time.)
# 
# Notice that `a` is `70` in this case and `b` is a pandas DataFrame.  In fact, it is the sub-DataFrame corresponding to the model_year of 70.  We would normally get this sub-DataFrame `b` using Boolean indexing, but here pandas `groupby` is doing that work for us.

# In[21]:


for a,b in df.groupby("model_year"):
    print(a)
    display(b)
    break # leave the for loop


# The above DataFrame `b` has length `29`.  The full DataFrame `df` has length `398`.

# In[22]:


len(df)


# Here is another example.  We tell the for loop to stop if we reach `a == 75`.  What I really want to see is, what is the value of `b` when `a == 75`.  If the `break` part is confusing, you can remove the `break` and just print "hi" or something, and you can then work with a later value of `b`.

# In[23]:


for a,b in df.groupby("model_year"):
    if a == 75:
        break


# Because we broke out of the for loop when `a` was equal to `75`, the value of `a` is still `75`.  It never reached the later values of `76` through `82`.

# In[24]:


a


# What I really want to emphasize is, what is the value of `b` when `a` is `75`?  It is exactly the sub-DataFrame of `df` corresponding to model_year 75.

# In[25]:


b


# Let's contrast that with how we would normally find this sub-DataFrame, using Boolean indexing.

# In[26]:


# Boolean indexing
sub_df = df[df["model_year"] == 75]


# I claim that the sub-DataFrame `sub_df` we just defined is the same as `b` that was produced above.  If we evaluate `sub_df == b`, we get a whole DataFrame of `True`s and `False`s.

# In[27]:


sub_df == b


# Are all of the values `True`?  (In other words, are the sub-DataFrames really equal?)  The following will tell us if all the values are equal along each individual row.

# In[28]:


(sub_df == b).all(axis=1)


# We can add another `all` to verify that all the rows really are equal.  The following confirms that the two DataFrames are equal.

# In[29]:


(sub_df == b).all(axis=1).all()


# Let's finally get to a question we can solve by iterating over a GroupBy object with a for loop.
# 
# * For each year, how many cars had mpg at least 25?
# 
# Here we change to better variable names than `a` and `b`.  We use the variable names `year` and `sub_df`.

# In[30]:


for year, sub_df in df.groupby("model_year"):
    print(year)


# We can't realistically print out every value of `sub_df`, but here is the final value.  (Inside of the for loop, `sub_df` takes a different value for each year from 70 to 82.  When we leave the for loop, `sub_df` is still the final value, the sub-DataFrame corresponding to model_year 82.)  Notice how the model_year is 82 throughout this DataFrame.

# In[31]:


sub_df


# Focusing just on this single sub-DataFrame, how can we determine how many values of "mpg" are greater than or equal to 25?  We won't be using Boolean indexing, but we will be using a Boolean Series.

# In[32]:


sub_df["mpg"] >= 25 


# Remember that Python treats `True` like `1` and treats `False` like `0`.  So we can count the number of `True` values just by adding up the values.

# In[33]:


(sub_df["mpg"] >= 25).sum()


# There was nothing special about this `sub_df` corresponding to model_year 82.  Within the for loop, we can use the exact same code.  For example, the results below say that in the sub-DataFrame corresponding to 75, there were only 5 cars with mpg of at least 25.

# In[35]:


for year, sub_df in df.groupby("model_year"):
    print(year)
    print((sub_df["mpg"] >= 25).sum())
    print()


# Let's try to make the output more readable using an f-string.  There is a very subtle mistake here.  Because my f-string uses double quotation marks `"` and my column name also uses double quotation marks, Python doesn't know which quotation marks go together.

# In[36]:


for year, sub_df in df.groupby("model_year"):
    print(f"In the year 19{year} there were {(sub_df["mpg"] >= 25).sum()} cars in the dataset with mpg at least 25.")


# Luckily this error is very easy to fix once we identify it.  We just switch one of the pairs of quotation marks to single apostrophes.

# In[38]:


for year, sub_df in df.groupby("model_year"):
    print(f"In the year 19{year} there were {(sub_df['mpg'] >= 25).sum()} cars in the dataset with mpg at least 25.")


# I think the code is more readable if we move the `(sub_df['mpg'] >= 25).sum()` portion onto its own line.

# In[39]:


# most important: understand what are year and sub_df as we move through this for loop
for year, sub_df in df.groupby("model_year"):
    num_cars = (sub_df['mpg'] >= 25).sum()
    print(f"In the year 19{year} there were {num_cars} cars in the dataset with mpg at least 25.")


# Notice how `num_cars` is `28`, the same value we saw above.

# In[40]:


num_cars


# Aside: there was a question about how `break` works.  Here is a simpler example (but it's very easy to be off slightly in your expectation).  Imagine `i` is `18`.  Then the condition in the if statement fails, so `i` increases to `21`, then the condition in the if statement is True, os we break out of the for loop.  So when we leave the for loop, `i` is `21`.

# In[41]:


for i in range(0,100,3):
    if i > 20:
        break


# In[42]:


i


# You'll be impressed at how easily we can switch from the above question, about absolute numbers, to the following question, about proportions.  The key reason behind this is again that `True` counts as `1` and `False` counts as `0`, so if we compute the `mean` of a collection of Trues and Falses, then that mean will represent exactly the proprtion of `True` values.
# 
# * For each year, what proportion of cars had mpg at least 25?

# In[43]:


# Because True is like 1 and False is like 0, can use mean to get the proportion
for year, sub_df in df.groupby("model_year"):
    prop_cars = (sub_df['mpg'] >= 25).mean()
    print(f"In the year 19{year}, {prop_cars} proportion of cars in the dataset had mpg at least 25.")


# Stare at the `prop_cars` code above until it makes sense.  For example, try it with a smaller Boolean Series, maybe a Boolean Series with three `True` values and one `False` value.  The `mean` of such a Series would be `0.75`.
# 
# The next two cells are just for fun, and are meant to show you some of the possibilities with string formatting.  Here the string formatting automatically converts from a proportion (between 0 and 1) to a percentage (between 0 and 100).

# In[45]:


# Because True is like 1 and False is like 0, can use mean to get the proportion
for year, sub_df in df.groupby("model_year"):
    prop_cars = (sub_df['mpg'] >= 25).mean()
    print(f"In the year 19{year}, {prop_cars:%} percent of cars in the dataset had mpg at least 25.")


# We can also make things look a little nicer by saying we only want two decimal places in the percentage.

# In[46]:


# Because True is like 1 and False is like 0, can use mean to get the proportion
for year, sub_df in df.groupby("model_year"):
    prop_cars = (sub_df['mpg'] >= 25).mean()
    print(f"In the year 19{year}, {prop_cars:.2%} percent of cars in the dataset had mpg at least 25.")


# We didn't get to any of the later portions of this notebook.

# ## Using the `mean` method of a pandas GroupBy object
# 
# The above for-loop method is very flexible, but for easier questions, often there are easier approaches.
# 
# * For each year, what was the average mpg?

# ## Visualizing the data
# 
# * Using Altair, make a scatter plot where the x-axis is the model_year, the y-axis is the mpg, and the color is encoded from the "origin" column.
# * Which of these fields (model_year, mpg, origin) could be most improved by specifying an encoding type?

# * How could we use `mean` and a bar chart to encode the average mpg data by year?  (Remove the color for this one.)

# * Put these last two Altair charts side by side using `alt.hconcat`.

# (There probably won't be time for this example.)
# 
# * What if we put the data into a facet chart, where we divide by weight?
