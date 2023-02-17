#!/usr/bin/env python
# coding: utf-8

# # Week 1 Wednesday
# 
# Most of today's class will be time to work on Worksheet 2 which is due Tuesday next week before discussion section.  Alicia, Chupeng, and I are here to help.
# 
# ## Announcements
# 
# * I have office hours in here (ALP 3600) today at 1pm.  I usually leave after about 20 minutes if nobody is around, so try to come before 1:20pm!
# * I added annotations to the Deepnote notebooks in the [course notes](https://christopherdavisuci.github.io/UCI-Math-10-F22/intro.html).  If you want to review what we covered on Monday or Tuesday, I recommend looking in the course notes instead of in Deepnote.  I don't know if I'll always find time to add annotations, but if there is a particular notebook you're stuck on, remind me on Ed Discussion to add annotations, or let me know what part is confusing.
# * No new material should be presented today or Thursday, so if you're feeling overwhelmed with how much has been introduced, this is a great chance to catch up.  By far the best way to learn this material is to try using it yourself.  That's a big reason for the worksheets.
# * Videos and video quizzes are due Friday before lecture.
# * Our first in-class quiz will be next week on Tuesday in discussion section.  It hasn't been written yet, but it will very likely include at least one question involving Boolean indexing and at least one question involving either the `str` accessor attribute or the `dt` accessor attribute.  The quizzes are closed book and closed computer.

# ## Warm-up
# 
# * In the attached vending machine file, what date occurs most often in the "Prcd Date" column?  (Don't worry about ties; just get the top date that occurs when using `value_counts`.)
# * What day of the week is that?

# In[2]:


import pandas as pd


# In[3]:


df = pd.read_csv("vend.csv")


# In[4]:


df["Prcd Date"]


# It's easy enough to look and find the most frequent value by looking at the pandas Series produced by calling `value_counts()`, but it is better to also have a "programmatic" way to get this value.

# In[6]:


myseries = df["Prcd Date"].value_counts()
myseries


# If we use `iloc`, that will get us the value, which is not what we want.  We want the corresponding key, which is `"3/30/2022"` in this case.

# In[11]:


myseries.iloc[0]


# I'm very surprised that the following works, instead of raising a key error.  My best guess is that, when the keys are strings, if you input an integer (without using `loc` or `iloc`), pandas defaults to using `iloc`.  But I'm not sure if that's the correct explanation, and at least for now, we should not use this kind of indexing with a pandas Series.

# In[7]:


myseries[0]


# This is the error I was expecting the previous cell to raise.

# In[8]:


myseries.loc[0]


# Here is the correct way to use `loc`.  Notice (by looking in the above series) how the value corresponding to this key is `2`.

# In[9]:


myseries.loc["8/14/2022"]


# Usually `loc` is not needed with a pandas Series, because you can just use square brackets directly to access a value by the key.

# In[10]:


myseries["8/14/2022"]


# Using `iloc` with this sort of key, or anything that is not an integer, will always raise an error.

# In[12]:


myseries.iloc["8/14/2022"]


# Here is a reminder of what `myseries` contains.

# In[13]:


myseries


# Let's finally get to accessing the most frequent date, 3/30/2022, using this series.  We can use the `index` attribute to get all of these keys.

# In[14]:


myseries.index


# The two most important pandas data types are the pandas Series and the pandas DataFrame data types.  There is also a pandas Index data type, but I think we can just pretend this is a list and everything will work fine.

# In[15]:


type(myseries.index)


# We want the initial element in this index, so we use indexing to access it.

# In[16]:


myseries.index[0]


# Now we can work on the second question, which was, what day of the week does this date correspond to?  This is harder to answer than on Tuesday, because the day of the week is not shown in the string.

# In[17]:


mydate = myseries.index[0]


# Notice that, for now, `mydate` really is a string.

# In[18]:


type(mydate)


# We'd rather this value be something that has functionality related to being a date, so we will convert it to a Timestamp using `pd.to_datetime`.

# In[19]:


pd.to_datetime(mydate)


# It really is a pandas Timestamp.

# In[20]:


type(pd.to_datetime(mydate))


# Now we can use the same `day_name` method we used on Tuesday.

# In[21]:


pd.to_datetime(mydate).day_name()


# If you don't remember the name of this method, you can always browse the options by using the Python `dir` function.

# In[22]:


dir(pd.to_datetime(mydate))

