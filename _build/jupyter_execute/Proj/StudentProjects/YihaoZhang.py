#!/usr/bin/env python
# coding: utf-8

# # Title

# Name: Yihao Zhang

# ID: 67069281

# ## Introduction

# The World Happiness Report is a landmark survey of the state of global happiness . The report continues to gain global recognition as governments, organizations and civil society increasingly use happiness indicators to inform their policy-making decisions. Leading experts across fields – economics, psychology, survey analysis, national statistics, health, public policy and more – describe how measurements of well-being can be used effectively to assess the progress of nations. 
# My project is to build a LinearRegression and predict Happiness score based on the variable of Economy, Familt, trust, health, and freedom,and find out which variabless might contribute the most to the model. At last I will try to use sklearn to determine whether a country's happiness is beyond expectation.

# In[1]:


import altair as alt
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np
from pandas.api.types import is_numeric_dtype
from sklearn.linear_model import LinearRegression


# ## Dataframe Outlook

# ### Import data：

# In[2]:


df = pd.read_csv("data happiness.csv", na_values=" ").dropna(axis=0).copy()


# In[3]:


df=df.loc[:,["Country name","Regional indicator","Ladder score","Logged GDP per capita","Social support","Healthy life expectancy","Freedom to make life choices","Generosity","Perceptions of corruption"]].copy()


# In[4]:


df


# In[5]:


df.shape


# Rank the Countries With Most Happiness Ladder Score

# In[6]:


df["Ladder score"].sort_values(ascending= False)


# In[7]:


df["Rank"]=pd.Series(range(1,150))


# The Daraframe is already in descending sorting, therefore we can distribute ranks to the countries directly.

# ### 10 Countries With Most Happiness Ladder Score

# In[8]:


df[["Country name","Rank"]].head(10)


# ### 10 Countries With Least Happiness Ladder Score

# In[9]:


df[["Country name","Rank"]].tail(10)


# ## Data visualization

# ### Distributing the Level of happiness regards to Quantiles of Ladder score

# In[10]:


Medianhp=df["Ladder score"].quantile(q=0.5)
firstq=df["Ladder score"].quantile(q=0.25)
thirdq=df["Ladder score"].quantile(q=0.75)


# In[11]:


def quartile(data):
    if data >= thirdq:
        return("Very Happy")
    if data <= firstq:
        return("Not Happy")
    if Medianhp >= data >= firstq:
        return("Less Happy")
    if thirdq >= data >= Medianhp:
        return("Happy")


# In[12]:


df["Happiness"]= df["Ladder score"].map(quartile)


# In[13]:


df[["Country name","Regional indicator","Happiness"]]


# Distrubute countries to 4 happiness level by using the 1st and 3 rd quantiles, and the median of the ladder score.

# ### Graph Regards to Level of Happiness

# In[14]:


c1 = alt.Chart(df).mark_point().encode(
    x=alt.X("Regional indicator", scale=alt.Scale(zero=False)),
    y=alt.Y("Ladder score", scale=alt.Scale(zero=False)),
    color="Happiness",
    tooltip=["Country name","Logged GDP per capita","Social support","Healthy life expectancy","Freedom to make life choices","Generosity","Perceptions of corruption"]
).properties(
    height=800,
    width=300,
    title="Country Region VS Happiness Ladder Score"
)
c1


# ### Ladder Score Global distrubution

# Use plotly to draw graph with world map to see the rough ladder score distribution related to the country region

# In[15]:


import plotly.express as px
fig = px.choropleth(
    df,
    locations="Country name",
    color="Ladder score",
    locationmode="country names",
    color_continuous_scale=px.colors.sequential.Plasma
)
fig.update_layout(title="Ladder Score worldwide distribution")
fig.show()


# In[16]:


import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
sns.kdeplot(data=df, x=df["Ladder score"], hue=df["Regional indicator"],fill=False, linewidth=2)
plt.axvline(df["Ladder score"].mean(), c="black")
plt.title("Ladder Score Distribution by Region")
plt.show()


# ### Scatterplot with Numerical variable as x axis, Ladder score as y axis, with Color Indicating its Region

# In[17]:


chart_list = []
for i in df.columns:
    c = alt.Chart(df).mark_circle().encode(
    x=alt.X(i, scale=alt.Scale(zero=False)),
    y = "Ladder score",
    color=alt.Color("Regional indicator", scale=alt.Scale(scheme='dark2')),
    tooltip=["Country name","Logged GDP per capita","Social support","Healthy life expectancy","Freedom to make life choices","Generosity","Perceptions of corruption"]
).properties(
    title=f"{i} VS Happiness Ladder Score"
)
    chart_list.append(c)


# In[18]:


alt.hconcat(chart_list[3], chart_list[4], chart_list[5],chart_list[6],chart_list[7],chart_list[8])


# Plotly Bar Chart with region as x axis and average ladder score as y axis

# In[19]:


import plotly.express as px
avg = pd.DataFrame(df.groupby('Regional indicator')['Ladder score'].mean())

fig = px.bar(df, x=avg.index, y=avg["Ladder score"])
fig.show()


# After drawing the diagram based on the Regional indicator, Happiness level and Ladder score. By looking at the chart above, we can say that Western Europe and Central and Eastern Europecountries tend to have happier perception of life, Sub-Saharan Africa and South Asia countries tend to have a less happier perception of current life. We can also see that economy, healthy life and social support form a clearer correlation with happiness ladder score compared to other variables.  We see a clearly positive correlation between GDP and Happiness, and between social support and Happiness

# ### Clustering the Countries use Standard Scaler and KMeans Clustering

# In[20]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# In[21]:


pipe = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(n_clusters=5))
    ]
)


# In[22]:


cols0=["Ladder score","Logged GDP per capita","Social support","Healthy life expectancy","Freedom to make life choices","Generosity","Perceptions of corruption"]


# In[23]:


pipe.fit(df[cols0])


# In[24]:


arr=pipe.predict(df[cols0])


# In[25]:


df["cluster"] = arr


# ### Scatterplot with Numerical variable as x axis, Ladder score as y axis, with Color Indicating its Cluster

# In[26]:


chart_list0 = []
for i in df.columns:
    e = alt.Chart(df).mark_circle().encode(
    x=alt.X(i, scale=alt.Scale(zero=False)),
    y = "Ladder score",
    color=alt.Color("cluster:N", scale=alt.Scale(scheme='blues')),
    tooltip=["Country name","Logged GDP per capita","Social support","Healthy life expectancy","Freedom to make life choices","Generosity","Perceptions of corruption"]
).properties(
    title=f"{i} VS Happiness Ladder Score with Cluster"
)
    chart_list0.append(e)


# In[27]:


alt.hconcat(chart_list0[3], chart_list0[4], chart_list0[5],chart_list0[6],chart_list0[7],chart_list0[8])


# ### Correlations and Feature Selection

# In[28]:


cols=["Logged GDP per capita","Social support","Healthy life expectancy","Freedom to make life choices","Generosity","Perceptions of corruption"]


# In[29]:


import matplotlib.pyplot  as plt
import seaborn as sns
plt.figure(figsize=(4,3))
sns.heatmap(df[cols].corr(),annot=True,fmt=".2f",linewidth=0.7)


# In[30]:


corr = df[df.columns].corr()
corr.sort_values(["Ladder score"], ascending = False, inplace = True)
print(corr["Ladder score"])


# We see the variable with most correlation with the ladder score is GDP, and the least correlated variable is Generosity.

# ## Linear Regression and Prediction

# ### Sklearn Linear Regression Based on Numerical Variables Only

# In[31]:


from sklearn.linear_model import LinearRegression
reg = LinearRegression()
cols=["Logged GDP per capita","Social support","Healthy life expectancy","Freedom to make life choices","Generosity","Perceptions of corruption"]
reg.fit(df[cols],df["Ladder score"])
pd.Series(reg.coef_,index=cols)


# In[32]:


reg.coef_


# In[33]:


print(f"The equation is: Pred Price = {cols[0]} x {reg.coef_[0]} + {cols[1]} x {reg.coef_[1]}+{cols[2]} x {reg.coef_[2]}+{cols[3]} x {reg.coef_[3]}+{cols[4]} x {reg.coef_[4]}+{cols[5]} x {reg.coef_[5]} + {reg.intercept_}") 


# In[34]:


df["Pred"] = reg.predict(df[cols])


# Constructing graph to see the differences of true ladder score and predicted data.

# In[35]:


import altair as alt
chartlist2=[]
for i in cols:
    c=alt.Chart(df).mark_circle().encode(
        x=alt.X(i, scale=alt.Scale(zero=False)),
        y="Ladder score",
        color=alt.Color("Regional indicator", scale=alt.Scale(scheme='dark2')),
    )
    c1=alt.Chart(df).mark_line().encode(
        x=alt.X(i, scale=alt.Scale(zero=False)),
        y="Pred",
    )
    c3=c+c1
    chartlist2.append(c3)
alt.hconcat(*chartlist2)


# ### Build training and test set：

# In[36]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(df[cols],df["Ladder score"],test_size = 0.3, random_state=1)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)
predtrain= model.predict(X_train)

predtest= model.predict(X_test)

print("Accuracy on Traing set: ",model.score(X_train,y_train))
print("Accuracy on Testing set: ",model.score(X_test,y_test))


# In[37]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
print("Mean Squared Error: ",mean_squared_error(y_test, predtest))
print("R^2 Score: ",r2_score(y_test,predtest))


# Interpreting the R^2 score, The model is explaining almost 67% of variablity of the variance fo the training data, and we get a score of 0.78 on the accuracy of Training set, a score of 0.67 on the accuracy of Testing set, 

# As we can see the model does not have a really good model score and accuracy, to improve our model, we put variable of region indicator into consideration.

# ## Improvements of Prediction Model

# ### Create Binary Variables Column to improve Model accuracy, taking consideration of country geographic location

# In[38]:


df2=pd.get_dummies(df["Regional indicator"])


# In[39]:


df3=pd.concat([df, df2], axis=1).copy()


# In[40]:


df3


# In[41]:


cols4=[ 'Logged GDP per capita', 'Social support', 'Healthy life expectancy',
       'Freedom to make life choices', 'Generosity',
       'Perceptions of corruption', 
       'Central and Eastern Europe', 'Commonwealth of Independent States',
       'East Asia', 'Latin America and Caribbean',
       'Middle East and North Africa', 'North America and ANZ', 'South Asia',
       'Southeast Asia', 'Sub-Saharan Africa', 'Western Europe']


# ### Build Training and Testing set for the New Model

# In[42]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(df3[cols4],df3["Ladder score"],test_size = 0.3, random_state=1)
from sklearn.linear_model import LinearRegression
model2 = LinearRegression()
model2.fit(X_train,y_train)
pred2= model2.predict(X_train)
predtest2=model2.predict(X_test)
print("Accuracy on Traing set: ",model2.score(X_train,y_train))
print("Accuracy on Testing set: ",model2.score(X_test,y_test))


# In[43]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
print("Mean Squared Error: ",mean_squared_error(y_test, predtest2))
print("R^2 Score: ",r2_score(y_test,predtest2))


# Interpreting the R^2 score, The new model is explaining almost 74% of variablity of the variance fo the training data, and we get higher model  accuracy score on both of the training set and testing set, and we also get less Mean Squared Error.

# ### sklearn Linear Regression of the New Model

# In[44]:


from sklearn.linear_model import LinearRegression
reg2 = LinearRegression()
reg2.fit(df3[cols4],df3["Ladder score"])
pd.Series(reg2.coef_,index=cols4)


# In[45]:


reg2.coef_


# In[46]:


print(f"The equation is: Pred Price = {cols4[0]} x {reg2.coef_[0]} + {cols4[1]} x {reg2.coef_[1]}+{cols4[2]} x {reg2.coef_[2]}+{cols4[3]} x {reg2.coef_[3]}+{cols4[4]} x {reg2.coef_[4]}+{cols4[5]} x {reg2.coef_[5]}+{cols4[6]} x {reg2.coef_[6]}+{cols4[7]} x {reg2.coef_[7]}+{cols4[8]} x {reg2.coef_[8]}+{cols4[9]} x {reg2.coef_[9]}+{cols4[10]} x {reg2.coef_[10]}+{cols4[11]} x {reg2.coef_[11]}+{cols4[12]} x {reg2.coef_[12]}+{cols4[13]} x {reg2.coef_[13]}+{cols4[14]} x {reg2.coef_[14]}+{cols4[15]} x {reg2.coef_[15]} + {reg2.intercept_}") 


# In[47]:


df3["Pred"] = reg2.predict(df3[cols4])
df3


# ### Ture Ladder Score VS. Predicted Ladder Score with New Model

# In[48]:


import altair as alt
chartlist3=[]
for i in cols:
    d=alt.Chart(df3).mark_circle().encode(
        x=alt.X(i, scale=alt.Scale(zero=False)),
        y="Ladder score",
        color=alt.Color("Regional indicator", scale=alt.Scale(scheme='dark2')),
    )
    d1=alt.Chart(df3).mark_line().encode(
        x=alt.X(i, scale=alt.Scale(zero=False)),
        y="Pred",
        color=alt.value("#FFAA00")
    )
    d3=d+d1
    chartlist3.append(d3)
alt.hconcat(*chartlist3)


# ## Use Sklearn Gradient Boosting Classifier to determine if a country's people's happiness is beyond expectation cosidering various variables.

# In[49]:


from sklearn.ensemble import GradientBoostingClassifier


# In[50]:


import numpy as np
conditions = [df3['Pred'] > df['Ladder score'],df3['Pred'] < df['Ladder score']]
choices = ['Under Expectation','Beyond Expectation']
df3['Happiness Expectation'] = np.select(conditions, choices, default='Beyond Expectation')
df3


# In[51]:


X_train,X_test,y_train,y_test = train_test_split(df3[cols4],df3['Happiness Expectation'],test_size = 0.3, random_state=1)


# In[52]:


clf = GradientBoostingClassifier(n_estimators=200, learning_rate=1.0,max_depth=1, random_state=0).fit(X_train, y_train)


# In[53]:


clf.score(X_test, y_test)


# The model score are too low,  indicate that the model cannot be used to determine whether a happiness score of one country is beyond expectation.

# # Summary

# I have completed a linear regression model to predict the happiness index of country residence. The results show that if all features are included in the dataset, the R squared of the model is about 73.5%. I compared the importance of each variable to the final output and found that GDP and Social support are the two most important factors. Generosity is the least important factor. The model based on the most dominant features has an accuracy of about 82.7% on the training dataset and about 73.5% on the test dataset. Building regional factors into my model improved the accuracy of my model. In addition, we performed machine learning to predict whether the happiness index will exceed expectations, but found that the data and models were insufficient for us to draw conclusions.

# # References

# Dataframe reference link: https://www.kaggle.com/datasets/ajaypalsinghlo/world-happiness-report-2021/code

# seaborn.kdeplot: https://seaborn.pydata.org/generated/seaborn.kdeplot.html

# plotly: https://plotly.com/python/plotly-express/

# Choropleth Maps in Python: https://plotly.com/python/choropleth-maps/

# Seaborn Heatmap: https://machinelearningknowledge.ai/seaborn-heatmap-using-sns-heatmap-with-examples-for-beginners/

# sklearn.ensemble.GradientBoostingClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html

# 

# 

# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=bf53c6e2-728a-4eab-a631-974d9b1a1438' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>
