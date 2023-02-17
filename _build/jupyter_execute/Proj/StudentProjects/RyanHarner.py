#!/usr/bin/env python
# coding: utf-8

# # Predicting Australian Cities With Weather
# 
# Author: Ryan Harner
# 
# email: ryanharner413@gmail.com
# 
# Course Project, UC Irvine, Math 10, F22

# ## Introduction
# 
# In this project, I will be looking at the data of Australian cities and their weather to attempt to predict a certain aspect of the dataset. I will be using Pipeline and StandardScaler, LinearRegression, PoissonRegressor, and Lasso to understand how "MaxTemp" is affected by the other parts of this dataset.  

# ## Main Portion of the Project

# Below I have all the libraries and modules needed for my project.

# In[1]:


import pandas as pd
import altair as alt
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures


# I used the weatherAUS.csv file from Kaggle.

# In[2]:


df = pd.read_csv("weatherAUS.csv")
df[:3]


# The columns are for the most part self-explainatory. The temperature columns are in Celcius, "Sunshine" is measured in hours, and other columns are measured with the metric system such as the "Evaporation" column with millimeters. Also the "Location" column is filled with Australian cities. If interested, check out [Reference](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package) to see the description of the columns.

# In[3]:


df.shape


# There's 145460 rows and 23 columns in df.
# To clean the data I choose to drop all the rows which had nan as an entry.

# In[4]:


df = df.dropna(axis=0)


# With is_numeric_dtype and list comprehension, I'm able to find the columns which have dtypes that are numerical. I told Xlist to keep only the first 6 using slicing.

# In[5]:


from pandas.api.types import is_numeric_dtype
Xlist = [c for c in df.columns if is_numeric_dtype(df[c]) == True][:6]
Xlist


# In[6]:


Xlist.append("Location") 
Xlist.append("Date") #adds Location and Date into the list
Xlist


# In[7]:


df_mini = df[Xlist] #creating DataFrame with strings in Xlist as columns


# Using list comprehension to see what the unique locations are. (under the stipulation that the name of the city is less than 7 letters)

# In[8]:


listcomp = [c for c in df_mini["Location"].unique() if len(c)<7] 
listcomp


# Boolean indexing allows us to shorten df_mini to a DataFrame with the entries for the "Location" column being the same as the strings in listcomp.

# In[9]:


df_mini = df_mini[df_mini["Location"].isin(listcomp)].copy()


# Using dtypes, I can see that the "Date" column has strings as entries. In following steps, I will make a new column "Datetime" that has datetime values as entries. Also I will drop the "Date" column and make df_mini have 5000 random rows.

# In[10]:


df_mini.dtypes


# In[11]:


datetimed = pd.to_datetime(df_mini["Date"]).to_frame() #This is a dataframe.


# In[12]:


df_mini["Datetime"] = datetimed["Date"] #method 1 to get series values into DataFrame
df_mini = df_mini.drop("Date",axis=1).sample(5000, random_state=82623597).copy()
df_mini[:3]


# In[13]:


df_mini.shape


# ## Graphs from Altair

# In[14]:


sel = alt.selection_single(fields=["Location"], bind="legend")
c1 = alt.Chart(df_mini).mark_circle().encode(
    x= "Datetime",
    y= "MaxTemp",
    color=alt.condition(sel,"Location", alt.value("grey")),
    opacity=alt.condition(sel, alt.value(1), alt.value(0.1)),
    tooltip=["Location","Datetime","MaxTemp"]
).properties(
    title='Max Temp Data'
).add_selection(sel)
c1


# For the two charts below this is how to make the chart interactive: (You can scroll your mouse to zoom in and out; left click and drag to move)

# In[15]:


sel = alt.selection_single(fields=["Location"], bind="legend")

c2 = alt.Chart(df_mini).mark_line().encode(
    x= "Datetime",
    y= "MaxTemp",
    color=alt.condition(sel,"Location", alt.value("grey")),
    opacity=alt.condition(sel, alt.value(0.65), alt.value(0.1)),
    tooltip=["Location","Datetime","MaxTemp"]
).add_selection(sel).interactive()
c2


# In[16]:


c1+c2


# For these graphs I focused on how the "MaxTemp" was changing over time with respect to the "Location". Looking at the graphs, I noticed that for some cities that there's long horizontal lines from lack of data. I talk more about this below in the caption for another graph, but to summarize, this is a result of how I cleaned the data with dropna().
# 
# Also there is definitely a pattern in these graphs. Although there isn't a positive or negative trend over the course of the timeframe, the points seem to make a zig-zagging pattern that seems to correspond to the month/season. 
# 

# Below I make a smaller dataframe consisting of only the rows which are in the year 2014. I do this because I eventually want to see how the columns affect "MaxTemp" within one year.

# In[17]:


df_mini = df_mini[(df_mini["Datetime"].dt.year==2014)].copy()


# In[18]:


sel = alt.selection_single(fields=["Location"], bind="legend")
c2 = alt.Chart(df_mini).mark_circle().encode(
    x= "Datetime",
    y= "MaxTemp", 
    color=alt.condition(sel, "Location", alt.value("grey")),
    opacity=alt.condition(sel, alt.value(1), alt.value(0.1)),
    tooltip=["Datetime"]
).add_selection(sel)
c2


# The graph's points are widely spread out, however they make a dipping pattern similar to a flattened out x^2 graph.
# 
# Creating new columns "Month" for df_mini which gives the month a number.

# In[19]:


df_mini["Month"]=df_mini["Datetime"].dt.month.values.copy()
df_mini
#method 2 to get series values into DataFrame


# In the following steps, I use groupby to get the averages for each month for columns "MaxTemp" and "MinTemp".

# In[20]:


df_mon = df_mini.groupby("Month").mean()[["MaxTemp","MinTemp"]]
df_mon


# In[21]:


print(f''' 
min MaxTemp: {min(df_mon["MaxTemp"])},  month: {(df_mon["MaxTemp"]).argmin()}
min MinTemp: {min(df_mon["MinTemp"])},  month: {(df_mon["MinTemp"]).argmin()}
''')


# In the above steps, by using groupby on the Month column of df_mini, we can see that the temperature for MaxTemp and MinTemp are lowest at month 6 and 7, respectively, which are June and July. These are winter months for Australia. This makes sense because winter is typically the coldest season.

# In[22]:


alt.Chart(df_mini,title="2016 Max Temperature (C) in Australian Cities").mark_rect().encode(
    x="Month:O",
    y="Location:O",
    color=alt.Color('MaxTemp:Q', scale=alt.Scale(scheme="inferno")),
    tooltip=[alt.Tooltip('MaxTemp:Q', title='Max Temp')]
).properties(width=550)


# Note that the colors correspond to "Max Temp" in degrees Celsius. The darker colors indicate a cooler temperature, and the warmer, yellow colors indicate that it is hot.
# 
# This colorful graph does more than just look pretty. It not only displays the temperatures of cities for each month in 2016, but it also tells a story about the data that was used. In the graph, there's missing data for the cities Sale and Hobart. This is from taking away the rows with nan values at the beginning of my project. From April to December, Sale has nan values in columns such as "Evaporation" making it get cut out of the df_mini dataset when I used dropna. Hobart also has similar things from January to April.
# 
# 
# Interpreting this map, we can also see that Hobart most likely is the coldest out of the cities since it has the lowest overall max temperature (C). However, Hobart's data will be affected since it is missing data from the summer. It is the most southern city (it is in Tasmania) so presumably the summers will be warmer than other cities and winters will be colder. Darwin would most likely be the hottest as its max temperature never drops below 27.3 degrees Celcius.
#  
# [Reference](https://altair-viz.github.io/gallery/weather_heatmap.html): Heatmap from Altair

# ## Standard Scaler
# 

# I create a list using list comprehension of all the column names that have numeric dtypes.
# I then add "Location" and remove "MaxTemp" as columns because I want to predict what the "MaxTemp" from the other columns.

# In[23]:


cols = [c for c in df_mini.columns if is_numeric_dtype(df_mini[c]) == True]
cols.remove("MaxTemp")
cols


# In[24]:


df_mini2 = df_mini.copy()


# Using StandardScaler() to fit then transform df_mini[cols] so that the mean of the columns (df_mini[cols]) is 0 and the standard deviation of the columns is 1. 

# In[25]:


scaler = StandardScaler() # mean=0 and std=1


# In[26]:


scaler.fit(df_mini[cols])


# In[27]:


df_mini2[cols] = scaler.transform(df_mini[cols])
df_mini2


# As seen below, for each column the mean is near 0 and the std is near 1.

# In[28]:


df_mini2[cols].mean()


# In[29]:


df_mini2[cols].std()


# ## Linear Regression
# Next we use LinearRegression(). We fit then predict. 

# In[30]:


reg = LinearRegression()


# In[31]:


reg.fit(df_mini2[cols], df_mini[["MaxTemp"]])


# Setting a column in df_mini2 called "pred" to be equal to the Linear Regression predict of df_mini2[cols].

# In[32]:


df_mini2["pred"] = reg.predict(df_mini2[cols])
df_mini2[:3]


# Below, I graph both the prediction and the "MaxTemp". It looks similar to the graph for "MaxTemp".

# In[33]:


c3 = alt.Chart(df_mini2).mark_line().encode(
    x= "Datetime",
    y= "pred", 
    tooltip=["Datetime"]
)


# In[34]:


c4 = alt.Chart(df_mini2).mark_line().encode(
    x= "Datetime",
    y= "MaxTemp", 
    tooltip=["Datetime"])
c3|c4


# ## Pipeline
# Pipeline is a way faster process of combining StandardScaler() and any type of regression. It requires a lot less code.

# In[35]:


pipe = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("reg", LinearRegression())
    ]
)


# In[36]:


pipe.fit(df_mini[cols],df_mini["MaxTemp"])


# In[37]:


pipe.predict(df_mini[cols])


# 3 cells of code is all that is needed to use Pipeline.
# Below, I inserted the predicted values into a column called "pred2" in df_mini2. 

# In[38]:


df_mini2["pred2"]=pipe.predict(df_mini[cols])


# Also, in the following line we can see that the "pred" column has all the same values as the "pred2" column in df_mini2. This is proof that Pipeline did the same thing as Standard Scaler first, then LinearRegression. 

# In[39]:


(df_mini2["pred"]==df_mini2["pred2"]).all()


# The coefficients and intercept are listed below. "MinTemp", "Sunshine", and "Evaporation" all have positive coefficients so I will use them to compare to cols and see how well they predict.  

# In[40]:


reg.coef_


# In[41]:


reg.coef_.shape


# In[42]:


pd.Series(reg.coef_.reshape(-1), index=reg.feature_names_in_)


# In[43]:


reg.intercept_


# The score tells us how well the prediction does. Closer to 1, the better.

# In[44]:


pipe.score(df_mini[cols],df_mini["MaxTemp"])


# ## PoissonRegressor
# Using PoissonRegressor() to predict "MaxTemp" with the cols list then with just "MinTemp", "Sunshine", and "Evaporation". 

# In[45]:


from sklearn.linear_model import PoissonRegressor


# In[46]:


pipe = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("pois", PoissonRegressor())
    ]
)


# In[47]:


pipe.fit(df_mini[cols],df_mini["MaxTemp"])


# In[48]:


df_mini2["pred3"] = pipe.predict(df_mini[cols])
df_mini2


# In[49]:


pipe.score(df_mini[cols],df_mini["MaxTemp"])


# Above, you can see that the score is lower when using Poisson Regressor than when I used Linear Regression.
# 
# Below, I am trying to use less columns to see if it affects the predict and score.

# In[50]:


pipe.fit(df_mini[["MinTemp","Evaporation","Sunshine"]],df_mini["MaxTemp"])


# In[51]:


pipe.predict(df_mini[["MinTemp","Evaporation","Sunshine"]])


# In[52]:


pipe.score(df_mini[["MinTemp","Evaporation","Sunshine"]],df_mini["MaxTemp"])


# The score after using Poisson Regression is about 0.05 less than when using Linear Regression. This means that Linear Regression is a better Regression model to use for the this data. A reasons why I assume this is the case is because in order to use Poisson Regression, it assumes that the variance is equal to the mean. We also said that the mean is zero when we did Standard Scaler.
# 
# Also when using less columns for the training data, the score goes down. This makes sense because intuitively using more data should give better results.
# 
# [Reference](https://scikit-learn.org/stable/auto_examples/linear_model/plot_poisson_regression_non_normal_loss.html)

# ## Lasso
# Trying Lasso to see if it works better.

# In[53]:


from sklearn.linear_model import Lasso


# In[54]:


pipe = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("lasso", Lasso())
    ]
)


# In[55]:


pipe.fit(df_mini[cols],df_mini["MaxTemp"])


# In[56]:


df_mini2["pred4"] = pipe.predict(df_mini[cols])
df_mini2


# In[57]:


pipe.score(df_mini[cols],df_mini["MaxTemp"])


# Out of all the Regression and linear models, Lasso worked the worst in terms of predicting "MaxTemp" using the columns from df_mini that were in the list cols.
# 
# [Reference](https://scikit-learn.org/stable/modules/linear_model.html#lasso)

# ## Summary
# 
# In the Altair section, I displayed charts of the "MaxTemp" in relation to time ("Datetime") and the cities ("Location"). In the machine learning section, I used Standard Scaler, Pipeline, Linear Regression, Poisson Regressor, and Lasso to predict the "MaxTemp". I showed that for my data, Linear Regression worked best and that using more columns allowed the predict and score to be better.

# ## References
# 
# Your code above should include references.  Here is some additional space for references.

# * What is the source of your dataset(s)?
# [Reference](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package)
# 
# 

# * List any other references that you found helpful.

# ## Submission
# 
# Using the Share button at the top right, **enable Comment privileges** for anyone with a link to the project. Then submit that link on Canvas.

# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=dede00de-9e69-4dc4-a447-70ffdf9c9c8f' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>
