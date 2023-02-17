#!/usr/bin/env python
# coding: utf-8

# # Week 1 Tuesday
# 
# Overall question for today (we will need some new techniques to answer it).
# * In the vending machines dataset, how many transactions occurred on Saturday?
# 
# We'll our solution into two parts.
# 1. How to determine if a string corresponds to Saturday.
# 2. How to determine how many elements in a pandas Series correspond to Saturday.

# In[2]:


import pandas as pd


# In[3]:


df = pd.read_csv("vend.csv")


# Let's remind our selves what this DataFrame looks like.  In the following, we've made a common mistake, and you'll be well-served to be able to recognize it.

# In[4]:


df.head


# If you look at the very first few words displayed in the output above, `<bound method ...`, that is a hint that we have forgotten to evaluate the function by using parentheses.  We need to use `df.head()` rather than `df.head`.  (The former, `df.head()`, is treating `head` as a method, which is correct.  The latter, `df.head`, is treating `head` as an attribute.)
# 
# In a Math 10 quiz or exam, you're not expected to have memorized when parentheses are necessary, but if you see an incorrect output like the above, you might be expected to recognize what has gone wrong.  And when completing the worksheets, it is very important to be able to correct these sorts of errors.
# 
# Here is a correct use of `df.head()`.

# In[5]:


df.head()


# Here is the full DataFrame.

# In[6]:


df


# ## Using string methods
# 
# Our overall goal is to count how many transactions occurred on Saturday.  We'll first see how to determine if a specific string represents Saturday, and then we'll see how to use the same approach on an entire pandas Series.
# 
# As humans, it's obvious to us that the following string represents a Saturday, but how can we programmatically get Python to recognize that?

# In[7]:


s = "Saturday, January 1, 2022"


# Scan through these string methods and attributes, skipping the "dunder methods" that begin and end with double underscores.  (In fact, you should just skip everything that starts with even one underscore.)  Do you see a method or attribute that could be used to determine if `s` contains the word "Saturday"?

# In[8]:


dir(s)


# A student suggested we try the `find` method, which is a great idea.  I don't know that method well, so let's check its help documentation.  The following doesn't work.

# In[9]:


help(find)


# The reason `help(find)` raised an error is because `find` is not defined independtly in Python, instead it is bound to this string object `s`.  So we need to instead call `help(s.find)`.  Notice how we *do not* use `help(s.find())`, because that would attempt to call `find` before we ever get to the `help` portion.

# In[10]:


help(s.find)


# We want to input the substring (denoted `sub` in the help documentation) that we want to find.  Here is a reminder of the value of the string `s`.

# In[11]:


s


# Now we call the `find` method of `s`. We don't need to use the optional arguments denoted by `start` and `end` in the documentation.  The following call outputs `0`, which is telling us that the substring was found at the very beginning (integer position 0) of `s`.

# In[12]:


s.find("Saturday")


# These string searches are case sensitive, so `"Day"` does not occur in `s`.

# In[13]:


s.find("Day")


# But `"day"` does occur, and it starts at integer position 5.

# In[14]:


s.find("day")


# If we're confident that `"Saturday"` will always occur at the beginning of the string, then another option is to use the `startswith` method.  Instead of returning a location, it returns a Boolean value of `True` or `False`.

# In[15]:


s.startswith("Saturday")


# It's important to remember that `True` is capitalized and that it does not use quotation marks (because it is not a string).

# In[16]:


True


# Let's see an example of calling this same method on a value taken directly from the DataFrame.  The following is a string, just like our `s` variable.

# In[17]:


df.loc[20, "TransDate"]


# This string does not start with `"Saturday"`, so the following outputs `False`.

# In[18]:


df.loc[20, "TransDate"].startswith("Saturday")


# ## Using string methods with a Series
# 
# We now know how to determine if an individual string contains `"Saturday"` or starts with `"Saturday"`.  How can we use that same approach on an entire pandas Series of strings?

# In[19]:


myseries = df["TransDate"]


# Here is the "TransDate" column.

# In[21]:


myseries


# When you ask Python, what is the type of a string, it outputs `str`.

# In[23]:


type("Saturday")


# Given a pandas Series containing strings, there is what is called an "accessor attribute" `str` which gives us access to Python string methods.  The type of `myseries.str` is not so important; I think I evaluated the following just to show that it is some special type of object, it's not a string and not a pandas Series.  (Much of Python is based on "Object Oriented Programming", and an effect of this is that Python libraries like pandas are filled with special objects that serve very specific purposes.)

# In[24]:


type(myseries.str)


# Notice how similar the attributes and methods listed here are to the methods and attributes defined on strings.  They are not exactly the same, but they're very similar.  (If you tried calling `dir` on a pandas Series, you would see almost no overlap between its methods and string methods, especially if you don't count the underscore methods.)

# In[22]:


dir(myseries.str)


# A quick reminder on slicing syntax.  We've only seen this for a DataFrame, but it works the same way for a Series.  Here we are getting the first four values of the pandas Series `myseries`.

# In[25]:


myseries[:4]


# This is as good of a time as any to mention the negative indexing notation.  Here is a way to get the last 4 values of something (whether a pandas Series as in this case, or a pandas DataFrame, or a Python list, etc).

# In[27]:


myseries[-4:]


# Using this string accessor attribute, `str`, we can use the `find` method, just like we used it above.  We see `0` at the top, because `"Saturday"` occurs starting at position `0` in the top few values, and we see `-1` at the bottom, because `"Saturday"` does not occur in the last few values, so the `find` call fails in that case.

# In[28]:


myseries.str.find("Saturday")


# Or we can use the `startswith` method.  Instead of returning a Series of integer values, this returns a series of Boolean values.

# In[26]:


myseries.str.startswith("Saturday")


# Our overall goal is to count, and that counting is somewhat more natural using the Boolean values.  The reason is because `True` behaves like `1` in Python (or in Matlab) and `False` behaves like `0`, so if we add up Boolean values, the result is the exact same as how many times `True` occurred.

# In[29]:


True+True+False


# Even though `True` and `1` are not literally the same object in Python, `True == 1` evaluates to `True`.

# In[30]:


True == 1


# A pandas Series, like `myseries.str.startswith("Saturday")`, has a `sum` method, which adds up all the values in the Series.  So the following is the first answer to our overall question, how many transaction dates correspond to Saturday?

# In[31]:


myseries.str.startswith("Saturday").sum()


# Here is another example solution, very similar, using Boolean indexing.  We first get the sub-DataFrame of `df` containing only those rows which correspond to a transaction that occurred on Saturday.  Notice how what goes inside the square brackets is our Boolean Series from above.

# In[32]:


# Boolean indexing
df[myseries.str.startswith("Saturday")]


# The answer to our question (which we already found above using the `sum` method) can now be found by determining the number of rows in the sub-DataFrame.  The sub-DataFrame has 684 rows and 18 columns.

# In[33]:


df[myseries.str.startswith("Saturday")].shape


# If we just want the number of rows, we can extract the number at position `0` from the `shape` tuple.

# In[34]:


df[myseries.str.startswith("Saturday")].shape[0]


# ## Using timestamp methods
# 
# It was easy to tell (at least as a human, if not as a computer) that our string `s` above corresponded to Saturday, because the word "Saturday" was in the string `s`.

# In[35]:


s


# A lot of times, dates will be provided to us in a more compact format.  For example, even in this same dataset, the dates in the "Prcd Date" column are written differently.  Here we access the date in the row labeled `20` and the column labeled `"Prcd Date"` using `.loc`.

# In[36]:


t = df.loc[20, "Prcd Date"]


# It's much more difficult to tell what day of the week this date corresponds to.

# In[37]:


t


# As a first step, we will use the `to_datetime` function in pandas to convert `t` into a Timestamp.  The following is not the correct syntax, because we have not told Python where to find the `to_datetime` function.

# In[38]:


# use the to_datetime function in pandas
to_datetime(t)


# The correct syntax to use is `pd.to_datetime`, which tells Python to get the `to_datetime` function from the pandas library.

# In[40]:


ts = pd.to_datetime(t)
ts


# The variable `t` is still a string.

# In[41]:


type(t)


# The variable `ts` that we just defined is a pandas Timestamp.  Because it is a new type of object that is specifically related to a time, it will have lots of new functionality.

# In[42]:


type(ts)


# Like usual, we can see all of the methods and attributes defined on `ts` by using Python's `dir` function.  Scroll through these and try to decide which ones will tell us that the date represents a Monday.

# In[43]:


dir(ts)


# A good guess is to use `day`, but that tells us the day of the month.  Because `ts` represents January 3rd 2022, the `day` attribute returns the integer `3`.

# In[44]:


ts.day


# Warning.  As far as I know, there is no good way to know in advance which of these are methods and which are attributes.  (And you certainly won't be tested on that.)  The important thing is to recognize when you have called one incorrectly.  Here we try calling `day` as a method.  The error says that "'int' object is not callable".  That is because `ts.day` is getting converted into `3`, so calling `ts.day()` is like calling `3()`, and that is the explanation for this error message.

# In[46]:


ts.day()


# Another try is `day_of_week`, and that is correct, but it's difficult to know in advance which day of the week corresponds to `0`.  Let's keep looking.

# In[47]:


ts.day_of_week


# The `dayofweek` attribute (note that there are no underscores) returns the same integer `0`.  We are looking for something that returns the string `"Monday"`.

# In[48]:


ts.dayofweek


# How about `day_name`?

# In[49]:


ts.day_name


# It looks like that didn't work.  This weird output is very similar (but shorter) to the `df.head` output we saw up above.
# 
# This output is a sign that we should be calling `day_name` as a method, not as an attribute.  The following works.

# In[50]:


ts.day_name()


# Aside.  You might wonder why I call `day_name()` a method instead of a function.  In general, if a function is bound to a Python object, like how this `day_name` is bound to `ts`, then it is usually called a method.  As small evidence of that, notice in the documentation how both the word "function" and the word "method" are used.

# In[51]:


help(ts.day_name)


# It's nice that we can get day of the week from a date.  If you had to implement that kind of functionality on its own, it would be very complicated.
# 
# Let's try it again.  Why doesn't the following work?

# In[52]:


"9/27/2022".day_name()


# The reason the above call to `day_name()` did not work, is because strings do not have a `day_name` method.  (Why would they?  What should `"Christopher".day_name()` return?)
# 
# To use this `day_name` method, we have to first convert the string into a Timestamp.

# In[53]:


pd.to_datetime("9/27/2022").day_name()


# ## Using timestamp methods with a Series
# 
# Let's return to our original question (which we already answered above using string methods) of how many transactions occurred on a Saturday.  Remember that the relevant column is the "TransDate" column.

# In[54]:


df["TransDate"]


# Just like we used `str` to access string methods on a pandas Series, here we use `dt` (which presumably stands for datetime) to access Timestamp methods on a pandas Series.  But we can't use `df["TransDate"].dt` directly.  The following mistake is basically the same as the `"9/27/2022".day_name()` mistake above. 

# In[55]:


dir(df["TransDate"].dt)


# We first need to convert from strings to Timestamps using `pd.to_datetime`.  Notice how the `dtype` changes.  (I'm not sure if there is a difference between the Timestamp objects we saw above and the type referred to below as `datetime64[ns]`.  I think it is safe to think of them as the same.)

# In[56]:


pd.to_datetime(df["TransDate"])


# Now we can call the Python `dir` function, like I tried to do above.  Notice how similar these methods are to the Timestamp attributes and methods we saw in the previous section.

# In[57]:


dir(pd.to_datetime(df["TransDate"]).dt)


# In particular, using this `dt` accessor attribute, we can call the `day_name` method.

# In[59]:


pd.to_datetime(df["TransDate"]).dt.day_name()


# Let's now see how often `"Saturday"` occurs.  Here is the same mistake we have made several times above.  We are missing the parentheses from `value_counts`.

# In[60]:


pd.to_datetime(df["TransDate"]).dt.day_name().value_counts


# Here is a correct call of `value_counts`.  This returns a Series.  The index of this Series contains the days of the week, and the values of this Series indicate how often they occur.

# In[61]:


pd.to_datetime(df["TransDate"]).dt.day_name().value_counts()


# To access the value corresponding to the key `"Saturday"`, we can use indexing.

# In[62]:


pd.to_datetime(df["TransDate"]).dt.day_name().value_counts()["Saturday"]


# We could also use the `.loc` indexing.

# In[63]:


pd.to_datetime(df["TransDate"]).dt.day_name().value_counts().loc["Saturday"]


# If we're willing to count to see what place `"Saturday"` occurs, we could also use the `iloc` indexing.  This `iloc` approach is definitely the worst approach in this case, but it is useful to know if you want to get, for example, the top value.

# In[64]:


pd.to_datetime(df["TransDate"]).dt.day_name().value_counts().iloc[5]

