#!/usr/bin/env python
# coding: utf-8

# # Global Life Expectancy
# 
# Author: Eden Kim
# 
# Course Project, UC Irvine, Math 10, F22

# ## Introduction
# 
# In this project, we will be looking at a historical dataset that contains the recorded life expectancy for years 1960 to 2020 from a number of unique coutries. Some things we want to explore is the country with the greatest life expectancy, life expectancy by region, and a global prediction line/trend.

# ## Country with Greatest Life Expectancy
# 

# First, we'll try to create a bar graph to visually find the country with greatest life expectancy.

# In[1]:


import pandas as pd
import altair as alt


# In[2]:


df = pd.read_csv("global life expectancy.csv")
df


# There are countries with missing data which we want to drop from the dataset because it will result in inaccurate calculations and problems in graphing.

# In[3]:


df1 = df.dropna(axis=0).copy()
df1


# Then, df.melt and groupby were used to get the average life expectancy for each country.
# `By using df.melt, I'm bringing all the years into one column. When I do groupby("Country").mean() on the resulting dataframe, it will take all the life expectancies corresponding to the country (for year 1960, 1961, ..., 2020) and take the mean.`

# In[4]:


df1_melt = df1.melt(
    id_vars=["Country Name", "Country Code"],
    var_name="Year",
    value_name="Life Expectancy"
)
df1_melt


# In[5]:


df1_bar = df1_melt.groupby("Country Name").mean().copy()
df1_bar.reset_index(inplace=True)
df1_bar


# Now we have the x-axis and y-axis data that we wanted. 
# To make the categorical bar graph, we use altair with df1_bar, "Country Name" as x-axis and "Life Expectancy" as y-axis.
# `I made the color in :Q so that we can distinguish which countries have higher and lower life expectancy. I also added the tooltip so that we can see the exact life expectancy value along with the country name.`

# In[6]:


alt.Chart(df1_bar).mark_bar().encode(
    x='Country Name',
    y='Life Expectancy',
    color='Life Expectancy:Q',
    tooltip=['Country Name','Life Expectancy']
)


# So this bar graph does show which countries have generally higher and lower life expectancies, but it's still hard to find which one of them has the greatest life expectancy on a glance.
# `We can get the country we're looking for by sorting the values of life expectancy from the dataset (df1_bar), finding the index of the max life expectancy and locating it on the dataframe.`

# In[7]:


df1_bar["Life Expectancy"].sort_values(ascending=False)


# In[8]:


df1_bar.loc[76,"Country Name"]


# With this, we have found that Iceland had the overall greatest life expectancy (of the countries in the data).

# ## Life Expectancy Over Time for Each Continent

# In this section, we want to see and compare the life expectancy trend (from 1960 to 2020) between different regions.
# As for how to divide the regions, I used continents.
# 
# In order to do this, I used a defined function that can return the continent given the country's name.
# The function is from https://stackoverflow.com/questions/55910004/get-continent-name-from-country-using-pycountry.
# Some things to note:
# - It required installing pycountry_convert module (which was moved to files: "requirements.txt")
# - There were some country names that couldn't be processed by the function because of the format or because the function didn't recognize the country as one. In order to address this, I used a try and except.
# - `Try: tries computing the following code and if any error occurs in the process, it's sent to the Except: where I handled the error by naming the continent as 'Misc'`

# In[9]:


import pycountry_convert as pc

def to_continent(country_name):
    country_alpha2 = pc.country_name_to_country_alpha2(country_name)
    country_continent_code = pc.country_alpha2_to_continent_code(country_alpha2)
    country_continent_name = pc.convert_continent_code_to_continent_name(country_continent_code)
    return country_continent_name


# In[10]:


continents = []
for i in range(len(df1)):
    try:
        continents.append(to_continent(df1.loc[i,"Country Name"]))
    except:
        continents.append('Misc')
continents


# We create a new column in the dataframe that has the continents.

# In[11]:


df2 = df1.copy()
df2["Continent"] = continents
df2


# Next, group the dataframe by continent so that we have year/life expectancy data by continent, not country.

# In[12]:


df_conti = df2.groupby("Continent").mean().copy()
df_conti.reset_index(inplace=True)
df_conti


# In order to make the line graph, we will use df.melt again to get the years into their own column. This time, the melt is used so that we can have the year column be the x-axis.

# In[13]:


df_melt = df_conti.melt(
    id_vars="Continent",
    var_name="Year",
    value_name="Life Expectancy"
)
df_melt


# Finally, we'll create a new altair chart, this time a line graph with "Year" as x-axis and "Life Expectancy" as y-axis, with 6 different lines for each continent (+ Misc).
# Since the lines are fairly close to each other, it's hard to distinguish, so I scaled it accordingly and made a selection portion as well.

# In[14]:


sel = alt.selection_single(fields=["Continent"], bind="legend")

alt.Chart(df_melt).mark_line().encode(
    x="Year",
    y=alt.Y("Life Expectancy", scale=alt.Scale(zero=False)),
    color=alt.condition(sel, "Continent", alt.value("lightgrey")),
    opacity=alt.condition(sel, alt.value(1), alt.value(0.2))
).add_selection(sel)


# With this, we can make some observations:
# - The overall trend for all continents is positive. (There is an increase in life expectancy over the years.)
# - Europe, North America, and Oceania (Australia) have almost the same trend, converging more as time passes.
# - There is a noticeable dip in the trend of South America, starting around 1987, lowest being around 1992.

# ## Global Trend in Life Expectancy (prediction line)

# In this section, we want to see th global trend and prediction line, using data from all of the countries.

# First we graph the scatter plot of life expectancy by year (took a sample because the data was too big).

# In[15]:


Life_Trend = alt.Chart(df1_melt.sample(500)).mark_point(size=20).encode(
    x = 'Year',
    y = 'Life Expectancy',
    color = alt.Color('Life Expectancy'),
    tooltip=['Country Name','Year','Life Expectancy']
).properties(
    width=800,height=300,
    title='Year x Life Expectancy'
)
Life_Trend


# Next, we make the linear regression line using scikit learn.

# In[16]:


from sklearn.linear_model import LinearRegression


# In[17]:


df_lin = df1_melt.copy()


# In[18]:


reg = LinearRegression()
reg.fit(df_lin[["Year"]], df_lin["Life Expectancy"])


# In[19]:


df_lin["Pred"] = reg.predict(df_lin[["Year"]])
df_lin


# We'll be using the prediction from regression to draw the regression/prediction line.

# In[20]:


alt.data_transformers.disable_max_rows()
c1 = alt.Chart(df_lin).mark_line().encode(
    x="Year",
    y="Pred"
)
Life_Trend+c1


# The line seems about right, fitted to the scatter plot.
# In order to find the trend, we'll want to calculate the slope of the line.
# We can find the slope using the x,y standard deviation, or with standardscaling.

# In[33]:


df_lin2.std(axis=0)


# In[34]:


slope = 5.332978/1.000044
slope


# This coefficient tells us that as time progressed, general life expectancy of people increased by approximately 5.33 years per year.

# ## Standardized Linear Regression

# While on the topic of linear regression I also wanted to try comparing it to regression rescaled to standard scale, and show that slope can be found using standard scaling as well.
# First off, we scale the data using StandardScaler.

# In[22]:


df_lin2 = df1_melt.copy()


# In[23]:


from sklearn.preprocessing import StandardScaler


# In[24]:


scaler = StandardScaler()
scaler.fit(df_lin2[["Year"]])


# In[25]:


df_lin2[["Year"]] = scaler.transform(df1_melt[["Year"]])
df_lin2


# We can check that the data was successsfully rescaled by seeing if the mean for the input/x-component ("Year") is close to 0, and std dev is close to 1.

# In[26]:


df_lin2.mean(axis=0)


# In[27]:


df_lin2.std(axis=0)


# Here, we'll make the second (but this time rescaled) regression line.

# In[28]:


reg2 = LinearRegression()
reg2.fit(df_lin2[["Year"]], df_lin2["Life Expectancy"])


# In[29]:


df_lin2["Pred"] = reg2.predict(df_lin2[["Year"]])
df_lin2


# In[30]:


reg2.coef_


# Here we can see that the coefficient refers to the slope and equals what we calculated before.

# In[36]:


c2 = alt.Chart(df_lin2).mark_line().encode(
    x="Year",
    y=alt.Y("Pred", scale=alt.Scale(zero=False))
)
c2


# ## Summary
# 
# In summary, I took a dataset from Kaggle that recorded the life expectancy of people in different countries from 1960 to 2020. Using the data, I first found which country had the overall greatest life expectancy, which was found to be Iceland. Then, I graphed the life expectancy trend by regions/continent which gave several observations, one of them which was that South America was an outlier with a noticeable dip unlike other continents around 1992. Lastly, I found the global trend/rate of increase in life expectancy using linear regression, which was found to be approximately 0.3 years per year.

# ## References
# 
# Your code above should include references.  Here is some additional space for references.

# * What is the source of your dataset(s)?
# 
# https://www.kaggle.com/datasets/hasibalmuzdadid/global-life-expectancy-historical-dataset

# * List any other references that you found helpful.
# 
# https://stackoverflow.com/questions/55910004/get-continent-name-from-country-using-pycountry

# ## Submission
# 
# Using the Share button at the top right, **enable Comment privileges** for anyone with a link to the project. Then submit that link on Canvas.

# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=9dc46a8c-2ea2-4012-b9fb-651ce1f184dd' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>
