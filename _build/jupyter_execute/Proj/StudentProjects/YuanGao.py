#!/usr/bin/env python
# coding: utf-8

# # Major League Baseball (MLB) Team Performances Predictions & Analysis
# 
# Author: Yuan Gao
# 
# Course Project, UC Irvine, Math 10, F22

# ## Introduction
# 
# Introduce your project here.  Maybe 3 sentences.

# In this project, I will explore the dataset of Major League Baseball (MLB), and as a baseball fan, I hope to find out some relationships between the team performances and those advances statistics. Also, I would like to predict the team performances. Last but not least, I would like to discover which factor(s) contribute the most to team performances.

# ## Main Portion of the Project
# 

# In[1]:


import pandas as pd
import altair as alt
import numpy as np
from sklearn.model_selection import train_test_split


# In[2]:


df = pd.read_csv("baseball.csv")
df


# I would like to change the column names from abbreviations to full names for better understanding. At the same time, I would like to change the numbers in columns which represent percentages from decimals to percentages by defining a new function.

# In[3]:


df = df.rename({"RS": "Runs Scored", "RA": "Runs Allowed", "W": "Wins", "OBP": "On-Base Percentage",
 "SLG": "Slugging Percentage", "BA": "Batting Average", "G": "Games Played", "OOBP": "Opponents On-Base Percentage", 
 "OSLG": "Opponents Slugging Percentage"}, axis="columns")


# In[4]:


df


# In[5]:


def to_percent(x):
    return round(x*100,2)


# In[6]:


cols = ["On-Base Percentage", "Slugging Percentage", "Batting Average", "Opponents On-Base Percentage", "Opponents Slugging Percentage"]
for x in cols:
    df[x] = df[x].map(to_percent)


# In[7]:


df


# In the years before 1999, there is no data of the opponents' statistics. Therefore, we would only use the data from 1999 till 2012.

# In[8]:


df = df[df["Year"] > 1998]
df


# In[9]:


df1 = df.copy()
df2 = df.copy()
df3 = df.copy()
df4 = df.copy()


# ## Predict Using the Decision Tree Classifier

# Since we want to find the relationships between team statistics and team performances (we evaluate team performances by whether the team enters the playoff or not), we would like to use the decision tree classifier to predict the team performances.

# In[10]:


del df1["RankSeason"]
del df1["RankPlayoffs"]
del df1["Year"]
del df1["Wins"]
del df1["Runs Scored"]
del df1["Runs Allowed"]
del df1["Games Played"]


# In[11]:


df1.columns


# In[12]:


from pandas.api.types import is_numeric_dtype
num_cols = [c for c in df1.columns if is_numeric_dtype(df[c])]
num_cols


# In[13]:


df1_1 = df[num_cols].drop("Playoffs", axis=1)
df1_1


# In[14]:


df1_1.isna().any(axis=0)


# In[15]:


df1_1.columns


# In[16]:


X_train, X_test, y_train, y_test = train_test_split(df1_1, df1["Playoffs"], test_size=0.2, random_state=0)


# In[17]:


from sklearn.tree import DecisionTreeClassifier


# In[18]:


for nodes in range(10,400,5):
    clf = DecisionTreeClassifier(max_depth=10, max_leaf_nodes=nodes)
    clf.fit(X_train, y_train)
    a = clf.score(X_train, y_train)
    b = clf.score(X_test, y_test)
    print(a,b)


# In the process above, I choose the max_leaf_nodes in the range (10,400) with an interval of 5. The result shows that the possibility that the train data predicts correctly is always around 0.99, and the possibility that the test data predicts correctly is always around 0.73 to 0.77. Therefore, there does not exist overfitting issue in the model above.

# In[19]:


clf_1 = DecisionTreeClassifier(max_depth=10, max_leaf_nodes=225)


# In[20]:


clf_1.fit(X_train, y_train)


# In[21]:


clf_1.score(X_train, y_train)


# In[22]:


clf_1.score(X_test, y_test)


# In[23]:


from sklearn.tree import plot_tree


# In[24]:


import matplotlib.pyplot as plt


# In[25]:


fig = plt.figure(figsize=(100,200))
plot_tree(
    clf_1,
    feature_names=clf_1.feature_names_in_,
    filled=True
);


# Since there's no overfitting issue in the model, I plan to plot one decision tree with a random number 225 as the max_leaf_nodes. We could discover that the clf_1.score is close to the clf.score above.

# ## Predict Using the Logistic Regression

# We have already used the DecisionTree Classifier to predict the possibility of entering playoff games. Therefore, I would like to use Logistic Regression to predict the possibility as well, and see if the results are similar or different.

# In[26]:


df1_1


# In[27]:


from sklearn.linear_model import LogisticRegression


# In[28]:


clf_2 = LogisticRegression()


# In[29]:


clf_2.fit(df1_1,df["Playoffs"])


# In[30]:


df2["Pred"] = clf_2.predict(df1_1)


# In[31]:


df2


# In[32]:


df2[df2["Playoffs"] == df2["Pred"]]


# In[33]:


358/420


# We could find out that in 420 rows of data, there are 358 rows which the prediction equals to the real statistic. We can see that there is nearly 85% of the correct predictions, which performs better than the results under DecisionTreeClassifier.

# In[34]:


from sklearn.metrics import mean_absolute_error


# In[35]:


mean_absolute_error(clf_2.predict(X_test), y_test)


# In[36]:


mean_absolute_error(clf_2.predict(X_train), y_train)


# We could discover that the error of the training set is larger than the error of the test set. Therefore, the results could be inaccurate.

# In[37]:


alt.data_transformers.enable('default', max_rows=500)

c = alt.Chart(df2).mark_rect().encode(
    x="Playoffs:N",
    y="Pred:N",
    color = alt.Color('count()', scale = alt.Scale(scheme = "redpurple", reverse = True)),
)

c_text = alt.Chart(df2).mark_text(color="black").encode(
    x="Playoffs:N",
    y="Pred:N",
    text="Pred"
)

(c+c_text).properties(
    height=200,
    width=200
)


# From the confusion matrix above, we could also verify our result above (85% correct predictions) as the two squares on the antidiagonal (wrong predictions) has the darkest color, which means that neither of them has counts over 50. 

# ## Figure Out Which Factor Is the Most Important 

# As a baseball fan, I understand the meaning and importance of all these statistics. However, I'm not sure which one(s) is the most important for teams to make playoffs. Thus, I would like to know which factor contributes the most to team performances.

# Since the method of Logistic Regression may not be accurate, I would use the test statistics from the DecisionTree Classifier method(clf_1).

# In[38]:


clf_1.feature_importances_


# In[39]:


pd.Series(clf_1.feature_importances_, index=df1_1.columns)


# In[40]:


df_fi = pd.DataFrame({"importance": clf_1.feature_importances_, "factors": clf_1.feature_names_in_})


# In[41]:


df_fi


# In[42]:


alt.Chart(df_fi).mark_bar().encode(
    x="factors",
    y="importance",
    tooltip=["importance"]
).properties(
    title = 'Importance of Factors'
)


# Thus, we could find out that the most important factor here is the opponents on-base percentage, which means that good pitchers and infielders with good defense could play the most important roles to help the team.

# In[43]:


c1=alt.Chart(df).mark_circle().encode(
    x=alt.X("Opponents On-Base Percentage", scale=alt.Scale(zero=False)),
    y=alt.Y("Slugging Percentage", scale=alt.Scale(zero=False)),
    color="Playoffs:Q",
    tooltip=["Team", "Year", "Opponents On-Base Percentage","Slugging Percentage"],
).interactive()

c1


# This graph clearly shows that the data points in the upper-left graph are more likely to make the playoffs since they have low opponents on-base percentage and high slugging percentage.

# In[71]:


import seaborn as sns


# In[80]:


sns.regplot(x=df1_1["Opponents On-Base Percentage"], y=df["Playoffs"], data=df2, logistic=True, ci=None, scatter_kws={'color': 'blue'}, line_kws={'color': 'red'})


# I also use the seaborn to draw a graph with logistic regression. I also use the most important factor (opponents on-base percentage) to model the predicted possibility of making the playoffs. It shows similar trends compared with the analysis above.

# ## Analysis of Offense and Defense

# Next, we would break up the data into offense part and defense part. For the offense, we would like to see the relationship between Runs Scored and other statistics (Batting Average, On-Base Percentage, Slugging Percentage). For the defense, we would like to see the relationship between Runs Saved and other statistics (Opponents On-Base Percentage, Opponents Slugging Percentage).

# In[44]:


df3


# ### Offense

# In[45]:


df3_of = df3[["On-Base Percentage", "Slugging Percentage", "Batting Average"]]
df3_of


# In[46]:


X_train, X_test, y_train, y_test = train_test_split(df3_of, df3["Runs Scored"], test_size=0.2, random_state=0)


# In[47]:


for nodes in range(10,400,5):
    clf_3 = DecisionTreeClassifier(max_depth=10, max_leaf_nodes=nodes)
    clf.fit(X_train, y_train)
    a = clf.score(X_train, y_train)
    b = clf.score(X_test, y_test)
    print(a,b)


# In[48]:


clf_4 = DecisionTreeClassifier(max_depth=10, max_leaf_nodes=225)


# In[49]:


clf_4.fit(X_train, y_train)


# In[50]:


pd.Series(clf_4.feature_importances_, index=df3_of.columns)


# Thus, we could discover that all three factors are nearly equally important, but the slugging percentage is more important, which is understandable since home runs are counted as slugs in the statistics.

# ### Defense

# In[51]:


df3_de = df3[["Opponents On-Base Percentage", "Opponents Slugging Percentage"]]


# In[52]:


X_train, X_test, y_train, y_test = train_test_split(df3_de, df3["Runs Allowed"], test_size=0.2, random_state=0)


# In[53]:


for nodes in range(10,400,5):
    clf_5 = DecisionTreeClassifier(max_depth=10, max_leaf_nodes=nodes)
    clf.fit(X_train, y_train)
    a = clf.score(X_train, y_train)
    b = clf.score(X_test, y_test)
    print(a,b)


# In[54]:


clf_6 = DecisionTreeClassifier(max_depth=10, max_leaf_nodes=225)


# In[55]:


clf_6.fit(X_train, y_train)


# In[56]:


pd.Series(clf_6.feature_importances_, index=df3_de.columns)


# We could discover that these two factors are almost equally important. However, this could only be used as a reference since this dataset doesn't provide us with the opponents' batting average.

# ## Los Angeles Dodgers Performance Analysis

# As a Dodgers fan, I would like to analyze the team's performance in the end. I would try to use some new methods here, such as StandardScaler and K-Neighbors Classifier.

# In[57]:


df4


# In[58]:


df_4 = df4.loc[df['Team'] == 'LAD']
df_4


# In[59]:


from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


# In[60]:


scaler = StandardScaler()
df_4[["Slugging Percentage","Opponents On-Base Percentage"]] = scaler.fit_transform(df_4[["Slugging Percentage","Opponents On-Base Percentage"]])

X_train, X_test, y_train, y_test = train_test_split(df_4[["Slugging Percentage","Opponents On-Base Percentage"]], df_4["Wins"], test_size=0.2)


# In[61]:


X = df_4[["Slugging Percentage", "Opponents On-Base Percentage"]]
X


# In[63]:


y = df_4["Wins"]
y


# We're randomly picking the n_neighbors=3.

# In[69]:


clf_7 = KNeighborsClassifier(n_neighbors=3)
clf_7.fit(X_train, y_train)
df_4["pred"] = clf_7.predict(df_4[["Slugging Percentage","Opponents On-Base Percentage"]])


# In[70]:


c2 = alt.Chart(df_4).mark_circle().encode(
    x="Opponents On-Base Percentage",
    y="Slugging Percentage",
    color=alt.Color("pred", title="Wins"),
    tooltip = ("Year", "Playoffs", "Slugging Percentage", "Opponents On-Base Percentage")
).properties(
    width=500,
    height=200,
)

c2


# Here, we can find out that the trend is similar to the graph c1 above. The data points on the upper-left could always make playoffs. However, the trend is not that obvious to wins since entering the playoffs or not also depends on the performances of other teams. Therefore, we might not get a clear conclusion here by just analyzing one team.

# ## Summary
# 

# In this project, we discover that using the Decision Tree Classifier is most accurate to make predictions than using the logistic regression. 
# 
# We also find out that slugging percentage and opponents on-base percentage are two most important factors affecting the teams' performances. We also find out that in teams' offense, slugging percentage is the most important factor. Thus, we suggest teams to find players who can hit hard as well as players who are good at defense.
# 
# However, we could not find a clear conclusion/relationship in teams' defense. Also, we cannot predict accurately whether a team enters the playoff or not by only analyzing the statistics of that team.

# ## References
# 
# Your code above should include references.  Here is some additional space for references.

# * What is the source of your dataset(s)?

# Kaggle.
# https://www.kaggle.com/datasets/wduckett/moneyball-mlb-stats-19622012

# * List any other references that you found helpful.

# https://www.datacamp.com/tutorial/k-nearest-neighbor-classification-scikit-learn
# I learned basic k-neighbors classifier here.
# 
# https://christopherdavisuci.github.io/UCI-Math-10-S22/Week7/Week7-Friday.html
# https://www.statology.org/plot-logistic-regression-in-python/
# I learned the logistic regression from here.
# 

# ## Submission
# 
# Using the Share button at the top right, **enable Comment privileges** for anyone with a link to the project. Then submit that link on Canvas.

# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=20db84bc-c87e-4591-bb66-31afb1efdb05' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>
