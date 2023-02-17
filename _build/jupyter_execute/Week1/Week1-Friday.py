#!/usr/bin/env python
# coding: utf-8

# # Week 1 Friday

# ## Announcements
# 
# * Videos and video quizzes due now.  Submit the quizzes sometime this weekend if you haven't finished them yet.
# * Our first in-class quiz is Tuesday during discussion section.  It is based on the material up to and including Worksheet 2 (especially Boolean indexing and the `str` and `dt` accessor attributes).

# ## Warm-up
# 
# One more class with the vending machines dataset.  We'll use a different dataset on Monday.

# In[3]:


import pandas as pd


# In[4]:


df = pd.read_csv("vend.csv")
date_series = df["TransDate"]


# In[5]:


date_series


# List each transaction date that corresponds to "Monday" in the DataFrame.  Don't list the same date twice.  Use the following strategies.

# * Use `find`, as on Worksheet 2.  (This approach was given on the Worksheet for practice, but I think you'll agree that the other two approaches given here feel more natural.  The `find` approach is most useful if we care about *where* the substring occurs.)

# It's good to be able to recognize the following error.  What do we need to add, so that there is a `find` attribute?

# In[6]:


date_series.find("Monday")


# We need to add the `str` string accessor attribute between the pandas Series and the `find` method.

# In[9]:


date_series.str.find("Monday")


# Recall that `-1` in this case means "not found".  We are interested in the positions where "Monday" *is* found, so we want to check that the value is *not* equal to `-1`.  This is a little indirect, and the below methods will feel more natural.

# In[8]:


date_series.str.find("Monday") != -1


# The above is an example of a "Boolean Series".  We now perform "Boolean indexing" using that Series.  This keeps all the rows in `df` where `True` occurs in the Series.

# In[10]:


df[date_series.str.find("Monday") != -1]


# We can also use Boolean indexing with a Series.  Here we are keeping only those values in `date_series` where the corresponding value is `True` in the Boolean index.  Phrased another way, the Boolean index tells us which values to keep in `date_series`.

# In[ ]:


date_series[date_series.str.find("Monday") != -1]


# We want to remove repetitions.  The most natural way to do this is with the `unique` method.

# In[12]:


date_series[date_series.str.find("Monday") != -1].unique()


# Just as an example, here we convert the above array into a Python list.

# In[13]:


list(date_series[date_series.str.find("Monday") != -1].unique())


# An alternative approach would be to convert the original Series (with the repetitions) into a set, because sets do not have repetitions.  (Two important things to memorize about a set are that they do not have repetitions, and that they are not ordered.)

# In[14]:


# sets do not have repetitions, so don't need unique in this part
set(date_series[date_series.str.find("Monday") != -1])


# * Use `contains`.
# 
# Here we will see what I think is a more natural approach.  This `contains` method is special, because it is not a method defined on strings themselves.

# In[15]:


dir('Monday, May 16, 2022')


# Here we verify that `"contains"` is not in the above list of string attributes and methods.

# In[16]:


"contains" in dir('Monday, May 16, 2022')


# On the other hand, `"contains"` is a method available on `date_series.str`.

# In[17]:


"contains" in dir(date_series.str)


# Aside: The main reason `contains` is not defined on strings is because you don't need that method, you can use the `in` operator instead.  The following says that `"stop"` appears as a substring in `"Christopher"`.

# In[18]:


"stop" in "Christopher"


# Here is a reminder of what `date_series` looks like.

# In[19]:


date_series


# We can get a Boolean Series corresponding to which of these strings contain the word `"January"`.  (We'll switch to `"Monday"` below.)

# In[20]:


date_series.str.contains("January")


# We can count the number of strings containing "January" by adding up all these `True` and `False` terms.  This sum will be the number of `True`s that occur, which is the same as the number of strings containing the word "January".

# In[21]:


date_series.str.contains("January").sum()


# Here we make two changes.  First, we switch from "January" to "Monday".  Second, we use Boolean indexing to get only the days in `date_series` which correspond to Monday.

# In[23]:


date_series[date_series.str.contains("Monday")]


# We can again get the unique values by calling the `unique` method.  I recommend comparing this code to the above code using `find`.  They are pretty similar in terms of length, but this code is easier to read, because the `contains` method immediately tells us `True` or `False`, unlike the `find` method which tells us an index.

# In[24]:


date_series[date_series.str.contains("Monday")].unique()


# * Use `day_name()`.
# 
# Here is another approach to the same problem, this time using datetime values.
# 
# Here is a common mistake.  The `dt` attribute cannot be called on `date_series`, which is a pandas Series of strings.

# In[25]:


date_series.dt.day_name()


# Instead, we need to first convert these strings to Timestamps, using `pd.to_datetime`.

# In[26]:


pd.to_datetime(date_series).dt.day_name()


# We now produce a Boolean Series indicating which days are Monday.

# In[27]:


# Boolean Series
pd.to_datetime(date_series).dt.day_name() == "Monday"


# We now can use Boolean indexing to get only the Mondays out of `date_series`.

# In[28]:


date_series[pd.to_datetime(date_series).dt.day_name() == "Monday"]


# We can again use the `unique` method to get rid of the repetitions.

# In[29]:


date_series[pd.to_datetime(date_series).dt.day_name() == "Monday"].unique()


# ## Which location had the lowest average sale price?
# 
# It's surprising how rarely in Math 10 we will write functions using `def` and how rarely we will use for loops, but these are essential parts of any programming language, and it is important to practice with them.

# * Write a function `ave_sale` which takes as input a location (like the string "GuttenPlans") and as output returns the average sale price (from the "RPrice" column) for transactions at that location in the vending machines dataset.

# We start out making this computation for a particular location.

# In[30]:


s = "GuttenPlans"


# Like above, here we are making a Boolean Series.  You could imagine that `df["Location"] == s` would just return `False`, since the Series on the left is not equal to the string on the right, but instead pandas uses what is called *broadcasting* to instead compare each individual value to `s`.  That is how we end up with the Boolean Series displayed below.

# In[31]:


df["Location"] == s


# Again, we can use Boolean indexing with the above Boolean Series.  We keep the rows corresponding to `True` in the above Boolean Series.  Notice how every location listed in this DataFrame is `"GuttenPlans"`.

# In[32]:


df[df["Location"] == s]


# From the DataFrame displayed above, we can now get the `"RPrice"` column.

# In[33]:


df[df["Location"] == s]["RPrice"]


# An alternative, that is equally good, is to start with the `"RPrice"` column, and then apply Boolean indexing to that.  This is the approach taken in the next cell.  It should create the exact same Series.

# In[34]:


df["RPrice"][df["Location"] == s]


# Once we have this Series, we can compute the average of the Series using the `mean` method.

# In[35]:


df["RPrice"][df["Location"] == s].mean()


# If you didn't know that the `mean` method existed, you could instead use the `sum` method and then divide by the length.  Make sure you are dividing by the length of this "GuttenPlans" Series, and not the length of the original DataFrame or the original full column `df["RPrice"]`.

# In[37]:


sub_series = df["RPrice"][df["Location"] == s]


# In[38]:


sub_series.sum()/len(sub_series)


# Now that we have seen how to get the desired answer for one particular location, it is easy to turn this into a general function.  (Be sure to practice writing this function syntax on your own.  It will be very difficult to remember unless you try writing it on your own, without looking at a sample.)  In this case, we are naming the input string `loc`, so we replace `s` in the above formula by `loc`.  The variable could just as well have been called something like `x` instead of `loc`.

# In[39]:


def ave_sale(loc):
    return df["RPrice"][df["Location"] == loc].mean()


# We get the same answer as above, which is good, but we should be careful to also test the function on other values.

# In[40]:


ave_sale("GuttenPlans")


# The fact that the function also works on another location, and gives us a distinct answer, is a good sign.

# In[41]:


ave_sale("Brunswick Sq Mall")


# * Define the same function, this time using a `lambda` function.
# 
# Notice how our answer above did not involve any intermediate compuations; the whole formula fit on a single line.

# In[42]:


# Full definition syntax
def ave_sale(loc):
    return df["RPrice"][df["Location"] == loc].mean()


# For that kind of short function definition, it is often more elegant to define it using what is called a `lambda` function.  In this case, the term `lambda` is telling Python that a function is going to be defined.  (This is like the `@` syntax in Matlab, for defining anonymous functions.)  The part that comes before the colon lists the zero or more input variables, and the part that comes after the colon is the returned value.  Notice how in the `def` notation, we need to use the `return` operator, but in this `lambda` function syntax, we do not explicitly use the word `return`.

# In[43]:


# lambda function definition
ave_sale2 = lambda loc: df["RPrice"][df["Location"] == loc].mean()


# Let's check that this `ave_sale2` gives the same answer as above.

# In[44]:


ave_sale2("Brunswick Sq Mall")


# * For each location, display the name of that location together with the average price.  Use a for loop.

# Here we are going to iterate through each unique location.

# In[45]:


df["Location"].unique()


# Aside: I wasn't sure if this was a NumPy array (more common) or a pandas array (less common).  It turns out, this output of the `unique` method is a NumPy array.

# In[46]:


type(df["Location"].unique())


# The fact that it is a NumPy array is not that important.  The important thing is that we can iterate through these values.  Here we just print them out.

# In[47]:


for loc in df["Location"].unique():
    print(loc)


# If we tried the same thing without calling `unique`, we would get over 6000 locations displayed, because repetitions are not being removed.

# In[48]:


for loc in df["Location"]:
    print(loc)


# There are more elegant ways to do this, but for now, we will print the location and then the corresponding average sale.

# In[49]:


for loc in df["Location"].unique():
    print(loc)
    print(ave_sale(loc))


# * Do the same thing, this time also using f-strings to display the information in a more readable format.
# 
# Postponed.  We'll see later, maybe in Week 2, a more elegant way to display the same information.

# * Put the same information into a dictionary, where the keys are the locations and where the values are the average sale prices.

# We start by making an empty dictionary.  (It's not obvious that `{}` should make an empty dictionary as opposed to an empty set.)

# In[50]:


d = {}


# Here we verify that `d` really is a dictionary.

# In[51]:


type(d)


# Even though we have `d` defined above, it's a good idea to put that definition into the same cell as our for loop, so that if we make a mistake, we can reset `d` just by evaluating this single cell.  Recall the syntax for setting a value in a Python dictionary: `d[key] = value`.

# In[52]:


d = {}
for loc in df["Location"].unique():
    d[loc] = ave_sale(loc)


# Here is the contents of `d`.

# In[53]:


d


# In a dictionary is a very convenient way to store this data, because we can access the values using indexing.  For example, `d["GuttenPlans"]` is equal to the average price of the sales at the GuttenPlans location.

# In[54]:


d["GuttenPlans"]


# * Which location had the lowest average sale price?  Answer this by converting the dictionary to a pandas Series, and then sorting the values and getting the zeroth index.

# Here is a reminder of what `d` looks like.

# In[55]:


d


# It really is a dictionary.

# In[56]:


type(d)


# Here is a first attempt to convert it to a dictionary.  Python does not know where to look for the definition of `Series`, because it is not a type defined in base Python.

# In[57]:


Series(d)


# Here again an error is raised, because this data type is case sensitive; it should be `pd.Series`.

# In[58]:


pd.series(d)


# This works.  Notice how the dictionary `d` has been converted into a pandas Series.

# In[59]:


pd.Series(d)


# If we want to find the location with the lowest average sale price, we can first sort the Series according to the values.  (By default, sorting is done in increasing order.)

# In[60]:


pd.Series(d).sort_values()


# We want the zeroth element in the corresponding index.  Here is the index.

# In[61]:


pd.Series(d).sort_values().index


# Here is the initial element in that index.  (This would not work without the `sort_values` call.)

# In[62]:


# lowest average sale price
pd.Series(d).sort_values().index[0]


# If instead you wanted the initial value, instead of the initial key, then you should use `iloc`.

# In[63]:


pd.Series(d).sort_values().iloc[0]

