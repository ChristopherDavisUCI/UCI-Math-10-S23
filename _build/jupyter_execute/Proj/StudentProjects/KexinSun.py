#!/usr/bin/env python
# coding: utf-8

# # Car Price Prediction

# Author: Kexin Sun

# Course Project, UC Irvine, Math 10, F22

# ## Introduction

# My project is to predict the price of future cars by analyzing the data in the data set. I first found out the brands with high sales volume, and then determined the factors affecting the price of the car by analyzing the correlation between various variables. I used Linear regression and train_test_split to analyze the prediction. I mainly analyzed the three most important factors: enginesize, curbweight, and horsepower to increase the accuracy of the model.

# ## Main portion of the project

# ### Data Loading

# In[1]:


import pandas as pd
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# In[2]:


df=pd.read_csv("CarPrice_Assignment.csv")
df


# In[3]:


df.shape


# In[4]:


df.isna().any().any()


# In[5]:


df["Brand"]=df["CarName"].apply(lambda x:x.split(" ")[0])
df["Brand"].unique()


# In[6]:


df=df.copy()


# In[7]:


df=df.replace(['alfa-romero','maxda','Nissan','porschce','toyouta','vokswagen','volkswagen'],['alfa-romeo','mazda','nissan','porsche','toyota','vw','vw'])


# In[8]:


df.drop("CarName",axis=1,inplace=True)


# ### Data Visualization

# In[9]:


c = alt.Chart(df).mark_bar().encode(
    x = "Brand",
    y = "count()",
    color="Brand"
).properties(title="Sales of Each Brand"
)
c


# According to the above chart, we can see that Toyota has the highest sales volume.

# In[10]:


avg=df.groupby("Brand")["price"].mean()
avg


# In[11]:


fig = px.bar(df, x=avg.index, y=avg)
fig.show()


# According to the above chart, we can see that the average selling price of Buick and Jaguar are higher than that of other brands.

# ### Find the correlation

# In[12]:


df.dtypes


# In[13]:


num_col = df.select_dtypes(exclude=['object']).columns
num = df[num_col].drop(['car_ID'],axis=1)


# In[14]:


cormatrix=num.corr()# find the correlationship among the dataset
cormatrix


# In[15]:


plt.figure(figsize = (20,20))
sns.heatmap(cormatrix, annot=True)
plt.show()


# From this graph we can see the correlation between the two variables.

# In[16]:


df["price"] = df["price"].astype(int)


# In[17]:


df2 = df.copy()
df2 = df2.merge(avg.reset_index(),how="left",on= "Brand")
bins = [0,10000,20000,40000]
label = ["cheap","ordinary","expensive"]
df["price_level"] = pd.cut(df2["price_y"],bins,right=False,labels=label)
df


# The above are car brands in the same price range based on their average price.

# In[18]:


cormatrix.sort_values(["price"], ascending = False, inplace = True)
print(cormatrix.price)


# We find a strong positive correlation between enginesize, curbweight, horsepower, carwidth, carlength and car price, while a strong negative correlation between citympg, highwaympg and car price.（Positive correlation: When one variable increases, another will also increase; Negative correlation: An increase in one variable and a decrease in another.）

# ### Create Linear Regression for the above 7 factors

# In[19]:


reg = LinearRegression()
cols=['curbweight','enginesize','horsepower','carwidth','carlength','citympg','highwaympg']
reg.fit(df[cols],df["price"])
pd.Series(reg.coef_,index=cols)


# In[20]:


reg.intercept_


# In[21]:


reg.coef_


# In[22]:


df["Pred1"] = reg.predict(df[cols])
df


# We find seven factors that affect the price and make price forecast, which are enginesize, curbweight, horsepower, carwidth, carlength, citympg and highwaympg.

# In[40]:


C=[]
for i in cols:
    c1=alt.Chart(df).mark_circle().encode(
        x=i,
        y="price",
        color="Brand"
    )
    C.append(c1)
alt.vconcat(*C)


# The configuration of this Altair chart was adapted from https://christopherdavisuci.github.io/UCI-Math-10-S22/Proj/StudentProjects/KehanLi.html.

# As can be seen from the figure above, enginesize, curbweight, horsepower, carwidth, carlength are positively correlated with price, while citympg, highwaympg are negatively correlated with price.

# In[24]:


c2 = alt.Chart(df).mark_circle().encode(
    x="enginesize",
    y="price",
    color="Brand"
)
c3=alt.Chart(df).mark_line(color="Blue").encode(
    x="enginesize",
    y='Pred1',
)
c2+c3


# ### Linear Regression for the most related 3 factor

# In[25]:


cols2=["enginesize","curbweight","horsepower"]


# In[26]:


reg2=LinearRegression()
reg2.fit(df[cols2],df["price"])
df["Pred2"]=reg2.predict(df[cols2])


# In[27]:


df


# In[28]:


reg2.intercept_


# In[29]:


reg2.coef_


# In[30]:


std= StandardScaler()
std.fit(df[cols2])
X= std.transform(df[cols2])


# In[31]:


y=df["price"]


# In[32]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.9, random_state=0)


# In[33]:


X.shape


# In[34]:


M=LinearRegression()
M.fit(X_train,y_train)
M.score(X_train, y_train)


# In[35]:


M.score(X_test, y_test)


# In[36]:


print(f''' 
accuracy on Train score: {M.score(X_train, y_train):.1%}
accuracy on Test score: {M.score(X_test, y_test):.1%}
''')


# In[37]:


pred=M.predict(X_test)


# In[38]:


mean_squared_error(y_test, pred)


# In[39]:


r2_score(y_test,pred)


# The model accounted for nearly 91% of the variance of the training data. But the score of testing data is too low, so it is not appropriate.

# ### Summary

# I built a linear regression model to predict future car prices. After comparing the influence of various factors on the final result, I identified three key factors, namely enginesize, curbweight and horsepower. It is also found that citympg, highwaympg and price will be negatively correlated. Based on the analysis of the three main factors, the results show that the accuracy of the model is nearly 91% on the training data set and about 50% on the test data set. The R square of the model is about 50%.

# ### Reference

# Dataframe: https://www.kaggle.com/datasets/gagandeep16/car-sales/code

# Plotly: https://plotly.com/python/plotly-express/

# seaborn heatmap: https://machinelearningknowledge.ai/seaborn-heatmap-using-sns-heatmap-with-examples-for-beginners/

# pandas.cut:https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.cut.html

# pandas.astype:  https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.astype.html

# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=f620a85d-d1ca-4f8c-a911-f5e4c03dc368' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>
