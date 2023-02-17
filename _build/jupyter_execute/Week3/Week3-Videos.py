#!/usr/bin/env python
# coding: utf-8

# # Week 3 Videos
# 
# * Which species in the penguins dataset has the longest median bill length?

# In[1]:


import pandas as pd
import altair as alt
import seaborn as sns


# In[2]:


df = sns.load_dataset("penguins")


# ## Median length using `groupby`
# 
# <iframe width="560" height="315" src="https://www.youtube.com/embed/nzmO5H7ueRo" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

# In[3]:


df.head(3)


# In[4]:


df.groupby("species")


# In[5]:


df.groupby("species").median()


# In[6]:


df.groupby("species").median()["bill_length_mm"]


# In[7]:


df.groupby("species").median()["bill_length_mm"].sort_values(ascending=False)


# In[8]:


df.groupby("species").median()["bill_length_mm"].sort_values(ascending=False).index


# In[9]:


df.groupby("species").median()["bill_length_mm"].sort_values(ascending=False).index[0]


# In[10]:


df.groupby("species").median()["bill_length_mm"]


# In[11]:


df.groupby("species").median()["bill_length_mm"].idxmax()


# ## Bar charts in Altair
# 
# <iframe width="560" height="315" src="https://www.youtube.com/embed/Y5di90SjU50" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

# In[30]:


alt.Chart(df).mark_bar().encode(
    x="species",
    y="bill_length_mm"
)


# In[31]:


df.sample(3)


# In[32]:


df.groupby("species").max()


# In[33]:


df.groupby("species").max(numeric_only=True)


# In[36]:


alt.Chart(df).mark_bar().encode(
    x="bill_depth_mm",
    y="bill_length_mm",
    color="species",
    tooltip=["bill_length_mm"]
)


# In[35]:


alt.Chart(df).mark_circle().encode(
    x="bill_depth_mm",
    y="bill_length_mm",
    color="species"
)


# ## Median values using Altair
# 
# <iframe width="560" height="315" src="https://www.youtube.com/embed/P1TAQiSCEcw" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

# In[37]:


alt.Chart(df).mark_bar().encode(
    x="species",
    y="bill_length_mm"
)


# In[39]:


alt.Chart(df).mark_bar().encode(
    x="species",
    y="median(bill_length_mm)",
    tooltip=["median(bill_length_mm)"]
)


# In[41]:


alt.Chart(df).mark_bar().encode(
    x=alt.X("species", sort="y"),
    y="median(bill_length_mm)",
    tooltip=["median(bill_length_mm)"]
)


# In[42]:


alt.Chart(df).mark_bar().encode(
    x=alt.X("species", sort="-y"),
    y="median(bill_length_mm)",
    tooltip=["median(bill_length_mm)"]
)


# In[43]:


my_index = df.groupby("species").median()["bill_length_mm"].sort_values(ascending=False).index
my_index


# In[44]:


alt.Chart(df).mark_bar().encode(
    x=alt.X("species", scale=alt.Scale(domain=my_index)),
    y="median(bill_length_mm)",
    tooltip=["median(bill_length_mm)"]
)


# In[45]:


my_list = list(my_index)
my_list


# In[46]:


alt.Chart(df).mark_bar().encode(
    x=alt.X("species", scale=alt.Scale(domain=my_list)),
    y="median(bill_length_mm)",
    tooltip=["median(bill_length_mm)"]
)

