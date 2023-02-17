#!/usr/bin/env python
# coding: utf-8

# # Week 4 Videos

# ## The `axis` keyword argument in pandas
# 
# <iframe width="560" height="315" src="https://www.youtube.com/embed/NIDqCZOSVB4" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

# ## The `axis` keyword argument: part 2
# 
# <iframe width="560" height="315" src="https://www.youtube.com/embed/UdCTan47Mfk" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


df = pd.DataFrame({
    "A": [2, np.nan, 3, 4],
    "B": [1, 5, 6, 2]
})


# In[ ]:


df


# In[ ]:


df.isna()


# In[ ]:


df.isna().any(axis=0)


# In[ ]:


df


# In[ ]:


df.dropna(axis=1)


# In[ ]:


df


# In[ ]:


df.rename({"B":"C"}, axis=1)


# In[ ]:


df.rename({2:"C"}, axis=0)


# In[ ]:


df.rename({2:"C"}, axis="rows")


# ## The pandas DataFrame method `apply`
# 
# <iframe width="560" height="315" src="https://www.youtube.com/embed/IlBZOGvS_G0" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

# In[1]:


import pandas as pd
import altair as alt


# In[ ]:


df = pd.read_csv("spotify_dataset.csv") # better: na_values = " "


# In[3]:


alt.Chart(df).mark_circle().encode(
    x="Energy",
    y="Loudness",
    color=alt.Color("Valence", scale=alt.Scale(scheme="spectral")),
    tooltip = ["Artist", "Song Name"]
)


# In[4]:


df


# In[5]:


df.dtypes


# In[6]:


pd.to_numeric(df["Energy"])


# In[7]:


import numpy as np


# In[8]:


df.applymap(lambda x: x if x != " " else np.nan)


# In[9]:


pd.to_numeric(df["Energy"])


# In[10]:


df = df.applymap(lambda x: x if x != " " else np.nan)


# In[11]:


pd.to_numeric(df["Energy"])


# In[12]:


df.dtypes


# In[13]:


lcol = "Popularity"
rcol = "Valence"


# In[15]:


df.loc[:, lcol:rcol] = df.loc[:, lcol:rcol].apply(pd.to_numeric, axis=0)


# In[16]:


df.dtypes


# In[17]:


alt.Chart(df).mark_circle().encode(
    x="Energy",
    y="Loudness",
    color=alt.Color("Valence", scale=alt.Scale(scheme="spectral")),
    tooltip = ["Artist", "Song Name"]
)


# In[18]:


alt.Chart(df).mark_circle().encode(
    x="Energy",
    y="Loudness",
    color=alt.Color("Valence", scale=alt.Scale(scheme="spectral", reverse=True)),
    tooltip = ["Artist", "Song Name"]
)


# In[19]:


df.loc[:, lcol:rcol]


# In[20]:


df.loc[:, lcol:rcol].sum(axis=0)


# In[21]:


df.loc[:, lcol:rcol].apply(lambda col: col.sum(), axis=0)


# In[22]:


df.loc[:, lcol:rcol].apply(lambda z: z.sum(), axis=0)


# In[23]:


df = pd.read_csv("spotify_dataset.csv") # better: na_values = " "


# In[24]:


df.replace(" ", np.nan, inplace=True)


# In[25]:


df = pd.read_csv("spotify_dataset.csv", na_values=" ")


# In[26]:


df.dtypes

