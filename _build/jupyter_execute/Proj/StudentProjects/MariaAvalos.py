#!/usr/bin/env python
# coding: utf-8

# # Analyzing and Visualizing Top Soccer Data
# 
# Author: Maria Avalos
# 
# Course Project, UC Irvine, Math 10, F22

# ## Introduction
# ![](european-leagues-1536x960.jpg)
# Soccer has widely been known to be a global sport. Even those who are not familiar with it may recognize the names of some of the greates players, such as Lionel Messi or Javier "Chicharito" Hernandez. Anyone who is familiar with the sport may also recognize what the top teams are of their respective league, such as F.C. Barcelona for the spanish leage 'La Liga' or Manchester City for the English leage 'the Premier League'. However what makes these teams to be able to constantly rank at the top? What could be the keys to their success.
# In this project, I explore data from the top European leagues to see what factors are affecting them the most to get theri high rankings using Regression models. I also explore where the top soccer teams are likely to score a goal to see get insight on success rates.

# ## Importing and Cleaning the Data

# In[1]:


import pandas as pd
import altair as alt
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from pandas.api.types import is_numeric_dtype
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


# In[2]:


df = pd.read_csv("EUClubSoccerStats.csv")
df


# In[3]:


#checking if contains missing values
df.isnull().values.any()


# Being that this is a pretty big DataFrame, it can be easy to miss but there are a good amount of missing values so we will clean it up. We will also drop the `Key` column as it won't be needed.

# In[4]:


df = df.dropna(axis=0).copy()
df = df.drop("Key", axis=1)
df.shape


# Since I want to investigate what factors effect a teams overall rank, I will first find the correlation among the dataset to the `Rank` and then create a sub-Dataframe containing those top factors as long as they are not repettive. Since `Rank` is from 1 being best to bigger number being worst, I have to check the 'negative' correlations as those will be the best. After finding it I will create my final Dataframe containing the averages stats of each team over the seasons to not overwhelm the data.

# In[5]:


df.corr()["Rank"].sort_values(ascending=True).head(60)


# In[6]:


df_sub = df.loc[:,['Team', 'League', 'Rank','Points', 'Wins', 'GoalDifference', 'GoalsPerGame','TotalAssistPerGame','OtherAssistPerGame', 'ShotsOnTargetPer90','Touches', 'TotalPassesPerGame','ShortPassesPerGame','TotalKeyPassesPerGame', 'DistMovedWithBall','TotalShotsPerGame','PassSuccess','Possession','SuccessfulDribblesPerGame']]
df_sub


# In[7]:


#taking average stats of every team for further analysis
df_avg = df_sub.groupby(["Team",'League'], sort=False, as_index=False).mean()
df_avg


# In[8]:


#example visualization with one of those top factors
alt.Chart(df_avg).mark_circle().encode(
    x= alt.X('Rank', scale=alt.Scale(domain=(1,21),reverse=True)),
    y='Wins',
    color="League:N",
    tooltip = ["Team", 'Rank','League']
)


# ## Predicting Factor Importance to Team Ranking

# Since I want to explore what factors are most important to a team to achieve higher ranking, in other words what factors make the top teams stay at the top,I decided to use regression to be able to find associations between these two things.

# ### Decision Tree Regression

# In[9]:


#getting the features we are going to use for predicition
features = [col for col in df_avg.columns if is_numeric_dtype(df_avg[col]) & (col!='Rank') & (col!='Team')& (col!='League')& (col!='Season')]
features


# In[10]:


X_train, X_test, y_train, y_test = train_test_split(df_avg[features], df_avg["Rank"], test_size=0.6, random_state=2868)


# As we are aware that decision trees can denote problems such as overfitting or underfitting,we can create a u-shaped error model to find the best number of leaf nodes. This configuration was adapted a [previous student]((https://christopherdavisuci.github.io/UCI-Math-10-S22/Proj/StudentProjects/TianyiChen.html#decision-tree)) and this [site](https://dziganto.github.io/cross-validation/data%20science/machine%20learning/model%20tuning/python/Model-Tuning-with-Validation-and-Cross-Validation/).

# In[11]:


train_error_dict = {}
test_error_dict = {}
for n in range(2,80):
    reg = DecisionTreeRegressor(max_leaf_nodes=n,random_state=2868)
    reg.fit(X_train, y_train)
    train_error_dict[n]= mean_squared_error(y_train, reg.predict(X_train))
    test_error_dict[n]= mean_squared_error(y_test, reg.predict(X_test))


# In[12]:


df_train = pd.DataFrame({"y":train_error_dict, "type": "train"})
df_test = pd.DataFrame({"y":test_error_dict, "type": "test"})
df_error = pd.concat([df_train, df_test]).reset_index()


# In[13]:


alt.Chart(df_error).mark_line(clip=True).encode(
    x="index:O",
    y="y",
    color="type"
)


# As we can see, n=6,7,8 are roughly where the best model is. Let's use this and hope our numbers look nice.

# In[14]:


reg = DecisionTreeRegressor(max_leaf_nodes=6)


# In[15]:


reg.fit(X_train, y_train)


# In[16]:


reg.score(X_train, y_train)


# In[17]:


reg.score(X_test, y_test)


# In[18]:


pd.Series(reg.feature_importances_, index=features).sort_values(ascending=False)


# Lots of zero values.

# In[19]:


df_importance = pd.DataFrame({"Importance": reg.feature_importances_, "Feature": reg.feature_names_in_})


# In[20]:


alt.Chart(df_importance).mark_bar().encode(
    x="Importance",
    y="Feature",
    tooltip=["Importance", "Feature"],
).properties(
    title="Importance of factors affecting Team Rankings",
    width = 900
)


# As we can see, this is not really that intersting. Of course realistically, it does make sense that the more `Wins` ones has, the higher their ranking. Let's try for bigger nodes to see if we get more intersting results just for fun.

# In[21]:


reg_fun = DecisionTreeRegressor(max_leaf_nodes=80)


# In[22]:


reg_fun.fit(X_train, y_train)


# In[23]:


reg_fun.score(X_train, y_train)


# In[24]:


reg_fun.score(X_test, y_test)


# In[25]:


pd.Series(reg_fun.feature_importances_, index=features).sort_values(ascending=False)


# In[26]:


df_importance = pd.DataFrame({"Importance": reg_fun.feature_importances_, "Feature": reg_fun.feature_names_in_})


# In[27]:


d_tree = alt.Chart(df_importance).mark_bar().encode(
    x="Importance",
    y="Feature",
    tooltip=["Importance", "Feature"],
).properties(
    title="Importance of factors affecting Team Ranking Using DecisionTree",
    width = 900
)
d_tree


# ### Random Forest Regression

# Random Forest regression can sometimes help with the problem of overfitting since they combine the output of multiple decision trees to come up with the final prediciton. So I've decided to test it out and see if I'd be given a different result. Let's check our error curve first.

# In[28]:


train_error_dict = {}
test_error_dict = {}
for n in range(2,25):
    rfe = RandomForestRegressor(n_estimators=1000,max_leaf_nodes=n,random_state=2868)
    rfe.fit(X_train, y_train)
    train_error_dict[n]= mean_squared_error(y_train, rfe.predict(X_train))
    test_error_dict[n]= mean_squared_error(y_test, rfe.predict(X_test))


# In[29]:


df_train = pd.DataFrame({"y":train_error_dict, "type": "train"})
df_test = pd.DataFrame({"y":test_error_dict, "type": "test"})
df_error = pd.concat([df_train, df_test]).reset_index()


# In[30]:


alt.Chart(df_error).mark_line(clip=True).encode(
    x="index:O",
    y="y",
    color="type"
)


# As we can see, we really only need `n` to be about 4 or 5 so we will have that in our first run.

# In[31]:


rfe = RandomForestRegressor(n_estimators=1000, max_leaf_nodes=5,random_state=2868)


# In[32]:


rfe.fit(X_train,y_train)


# In[33]:


rfe.score(X_train,y_train)


# In[34]:


rfe.score(X_test,y_test)


# In[35]:


df_importance1 = pd.DataFrame({"importance": rfe.feature_importances_, "feature": rfe.feature_names_in_})


# In[36]:


pd.Series(rfe.feature_importances_, index=features).sort_values(ascending=False)


# We can already see better number results (no zero values).

# In[37]:


rand_for = alt.Chart(df_importance1).mark_bar().encode(
    x="importance",
    y="feature",
    tooltip=["importance", "feature"],
).properties(
    title="Importance of factors affecting Blue's win using RandomForest",
    width = 900
)
rand_for


# In[38]:


#comparison between the twoo
d_tree|rand_for


# Overall, it is safe to conclude that based on the feature importances that I did to the data, `Wins` clearly are what affect a team's ranking the most followed by `Points` and `GoalDifference`.

# ## Visualizing Top Team Insights

# As we saw from my results, it appears that the amount of `Wins` a team gets, on average, throughout their season. But since soccer is a overall sport where one thing is affected by another (ex:rank is affected by wins which is affected by goals scored), I wanted to analyze where the Top Teams of two different leagues are more likely to score in hopes of demonstrating where one should attempt to score to get more points or getting some type of insight on their key to success.

# ### Shot Map

# Here I will create two DataFrames containing the top 10 teams from La Liga, the Spanish league, and the Premier League, the English league.

# In[39]:


liga = df[(df['League'] == 'La Liga') & (df['Rank']<= 10)]
liga = liga.loc[:,['Team','Rank','SixYardGoalsPerGame','PenaltyAreaGoalsPerGame','OutOfBoxGoalsPerGame']]
liga


# In[40]:


premier = df[(df['League'] == 'Premier League') & (df['Rank']<= 10)]
premier = premier.loc[:,['Team','Rank','SixYardGoalsPerGame','PenaltyAreaGoalsPerGame','OutOfBoxGoalsPerGame']]
premier


# Unfortunately, this data did not come with the coordinates to plot where these shots were taken for our [shot map](https://towardsdatascience.com/how-to-analyze-football-event-data-using-python-2f4070d551ff) we will have to create our own. Luckily, there are statistics on where on the field the team averaged to score per game so we can create a rough estimate.

# In[41]:


#points for 'La Liga' teams
y_s = [random.randint(114,120) for i in range(50)]
liga['y_s'] = y_s
x_s= [random.randint(30,50) for i in range(50)]
liga['x_s'] = x_s
y_p = [random.randint(104,114) for i in range(50)]
liga['y_p'] = y_p
x_p = [random.randint(0,80) for i in range(50)]
liga['x_p'] = x_p
y_o = [random.randint(80,100) for i in range(50)]
liga['y_o'] = y_o
x_o = [random.randint(0,80) for i in range(50)]
liga['x_o'] = x_o


# In[42]:


#points for 'Premier League' Teams
y_s = [random.randint(114,120) for i in range(50)]
premier['y_s'] = y_s
x_s= [random.randint(30,50) for i in range(50)]
premier['x_s'] = x_s
y_p = [random.randint(104,114) for i in range(50)]
premier['y_p'] = y_p
x_p = [random.randint(0,80) for i in range(50)]
premier['x_p'] = x_p
y_o = [random.randint(80,100) for i in range(50)]
premier['y_o'] = y_o
x_o = [random.randint(0,80) for i in range(50)]
premier['x_o'] = x_o


# In[43]:


#size of points to be the amount of times they scored
size_s = liga['SixYardGoalsPerGame'].to_numpy()
s_s = [1000*s_s**2 for s_s in size_s]
size_p = liga['PenaltyAreaGoalsPerGame'].to_numpy()
s_p = [1000*s_p**2 for s_p in size_p]
size_o = liga['OutOfBoxGoalsPerGame'].to_numpy()
s_o = [1000*s_o**2 for s_o in size_o]


# In[44]:


size_s = premier['SixYardGoalsPerGame'].to_numpy()
s_s = [1000*s_s**2 for s_s in size_s]
size_p = premier['PenaltyAreaGoalsPerGame'].to_numpy()
s_p = [1000*s_p**2 for s_p in size_p]
size_o = premier['OutOfBoxGoalsPerGame'].to_numpy()
s_o = [1000*s_o**2 for s_o in size_o]


# Looking at analysis that have been done with soccer statistics I found [this website](https://mplsoccer.readthedocs.io/en/latest/gallery/pitch_setup/plot_pitches.html) that has a program that assists with creating visualizations so that is what I will use.

# In[45]:


#install the program for graphics
get_ipython().system('pip install mplsoccer==1.1.9')


# In[46]:


from mplsoccer.pitch import VerticalPitch


# In[47]:


pitch = VerticalPitch(half=True,pitch_color='grass', line_color='white', stripe=True)
fig, ax = pitch.draw()
plt.gca().invert_yaxis()
#scatter plot of goal locations
plt.scatter(liga['x_s'],liga['y_s'],s=s_s, c = "#800020",alpha=0.5)
plt.scatter(liga['x_p'],liga['y_p'],s=s_p, c = "#FFD700",alpha=0.5)
plt.scatter(liga['x_o'],liga['y_o'],s=s_o, c = "#0000FF",alpha=0.5)


# In[48]:


pitch = VerticalPitch(half=True,pitch_color='grass', line_color='white', stripe=True)
fig, ax = pitch.draw()
plt.gca().invert_yaxis()
#scatter plot of goal locations
plt.scatter(premier['x_s'],premier['y_s'],s=s_s, c = "#800020",alpha=0.5)
plt.scatter(premier['x_p'],premier['y_p'],s=s_p, c = "#FFD700",alpha=0.5)
plt.scatter(premier['x_o'],premier['y_o'],s=s_o, c = "#0000FF",alpha=0.5)


# Both top teams from both leagues seem to have the most success making a goal if they shoot anywhere in the penalty area distance as this is wheere the points are the biggest.

# ### Heat Map

# Another visualization that can help us is a heat map to see where the most activity is happening. Let's try.

# In[49]:


#getting the goal point locations only
x_points = liga.melt(value_vars = ['x_s','x_p','x_o'])
y_points = liga.melt(value_vars = ['y_s','y_p','y_o'])
liga_pts = pd.DataFrame({'x_points':x_points['value'],'y_points':y_points['value']})


# In[50]:


x_points = premier.melt(value_vars = ['x_s','x_p','x_o'])
y_points = premier.melt(value_vars = ['y_s','y_p','y_o'])
premier_pts = pd.DataFrame({'x_points':x_points['value'],'y_points':y_points['value']})


# In[51]:


pitch = VerticalPitch(half=True,pitch_color='grass', line_color='white')
fig, ax = pitch.draw()
plt.gca().invert_yaxis()
#heat map
kde1 = sns.kdeplot(
    x = liga_pts['x_points'],
    y = liga_pts['y_points'],
    levels=20,
    fill=True,
    alpha=0.6,
    cmap = 'magma'
)
kde1


# In[52]:


pitch = VerticalPitch(half=True,pitch_color='grass', line_color='white')
fig, ax = pitch.draw()
plt.gca().invert_yaxis()
#heat map
kde2 = sns.kdeplot(
    x = premier_pts['x_points'],
    y = premier_pts['y_points'],
    levels=20,
    fill=True,
    alpha=0.6,
    cmap = 'magma'
)


# Proving our initial scatter plot conclusion, here we can see that indeed, most of the activity when scoring goals happens `Six Yards` from the goal post and the `Penalty Area`.

# ## Summary

# After testing using two different models, DecisionTree Regression and RandomForest Regression, I was able to get an insight on what are the most important factors that affect a teams rankings. Clearly the top teams are the top teams because they Win and Score the most. Goal difference is also important, which makes sense as scoring the most is not important if you are getting scored on just as much, cancelling out any progress.
# I was also able to conclude, from my visualizations, that the tops teams score the most when they shoot in the 6-yard area the most followed by the penalty area.

# ## References
# 
# Your code above should include references.  Here is some additional space for references.

# * What is the source of your dataset(s)?

# Source : [Performance Data on Football teams 09 to 22](https://www.kaggle.com/datasets/gurpreetsdeol/performance-data-on-football-teams-09-to-22?topic=trendingDataset&sort=votes&page=6)

# * List any other references that you found helpful.

# * [Shot Map Inspiration](https://towardsdatascience.com/how-to-analyze-football-event-data-using-python-2f4070d551ff)
# * [Shot Map python Tutorial](https://www.youtube.com/watch?v=2RhTuRWNqUc&t=634s)
# * [Matplotlib soccer program](https://mplsoccer.readthedocs.io/en/latest/gallery/pitch_setup/plot_pitches.html#sphx-glr-gallery-pitch-setup-plot-pitches-py)
# * [Random Number List](https://www.tutorialspoint.com/generating-random-number-list-in-python)
# * [Changing Scatterplot Marker Size](https://stackabuse.com/matplotlib-change-scatter-plot-marker-size/)
# * [Seaborn KDEPlot](https://seaborn.pydata.org/generated/seaborn.kdeplot.html)
# * [U test model](https://christopherdavisuci.github.io/UCI-Math-10-S22/Proj/StudentProjects/TianyiChen.html#summary)
# * [Also U test model](https://dziganto.github.io/cross-validation/data%20science/machine%20learning/model%20tuning/python/Model-Tuning-with-Validation-and-Cross-Validation/)
# * [Understanding Correlation](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.corr.html)
# * [Feature Importance](https://machinelearningmastery.com/calculate-feature-importance-with-python/)
# * [Random Forest Inspiration/Help](https://medium.com/@nicholasutikal/predict-football-results-with-random-forest-c3e6f6e2ee58)
# * [Decision Tree Regression Help](https://towardsdatascience.com/machine-learning-basics-decision-tree-regression-1d73ea003fda)
# 
# 

# ## Submission
# 
# Using the Share button at the top right, **enable Comment privileges** for anyone with a link to the project. Then submit that link on Canvas.

# In[52]:





# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=3e6de7f7-b97c-48f2-8ed1-3ba95424bab6' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>
