#!/usr/bin/env python
# coding: utf-8

# # Key Financial Metrics Analysis of S&P 500 Stocks
# 
# Author: Leander von Schönfeld
# 
# Email: leander.von.schoenfeld@studium.uni-hamburg.de
# 
# LinkedIn: [https://www.linkedin.com/in/leander-von-schoenfeld/](https://www.linkedin.com/in/leander-von-schoenfeld/)
# 
# Course Project, UC Irvine, Math 10, F22

# ## Introduction
# 
# With this project I want to analyze different properties of the S&P 500 stock index using stock data from the last 12 years and current valuation multiples. The main topics will be to evaluate whether you can classify the sector of a stock using key financial metrics, to predict the five year movement of a stock using valuation multiples and vice versa and a small time series analysis predicting stock prices of stocks from the utilities sector.

# ## Loading our datasets and cleaning them up

# In[1]:


import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import seaborn as sns
import altair as alt
import numpy as np


# First we import our datasets and get a feeling for the data by looking at their first rows.

# In[2]:


prices=pd.read_csv("sp500_stocks.csv")
prices.head()


# In[3]:


multiples=pd.read_csv("financials.csv")
multiples.head()


# We want to find all companies, where the data is incomplete, and get rid of these companies in our dataset.

# In[4]:


miss_val=multiples[multiples.isna().any(axis=1)]["Symbol"]
miss_val


# In[5]:


multiples_true = multiples.dropna()
multiples_true


# In[6]:


len(miss_val)==len(multiples)-len(multiples_true)


# As we see there are 10 companies with missing data. We check that dropna works correctly, by seeing if the number of rows in our new dataframe are the same.

# Next we only focus on Adjusted Closing Dates in our prices dataset because we look for longterm trends and in the difference between the adjusted closing price and the other prices is quite small on average. Therefore we drop all other rows.

# In[7]:


prices_adj = prices.drop(["Close", "High", "Low", "Open", "Volume"], axis=1).dropna()
prices_adj.head()


# Next we want to get rid of all stocks, where some of the data about the financial statements is missing and therefore drop these rows.

# In[8]:


prices_true=prices_adj[~prices_adj["Symbol"].isin(miss_val)]
prices_true


# ## Plotting the S&P 500 index normed at year 2010

# We want to get a feeling for how the S&P 500 developed over the years, using just our data. We add a year column but also put the symbol into the year column so that we can still match the data of the respective stocks together.

# In[9]:


prices_sectors = prices_true.copy()
prices_sectors["Year"] = prices_sectors.Date.str[0:4]+ ' ' + prices_sectors['Symbol']
prices_sectors.head()


# To see an overall trend we use the first stock price of every stock and every year.

# In[10]:


prices_sectors.drop_duplicates(subset ='Year', keep ='first', inplace =True)
prices_sectors.drop(["Date"],axis=1,inplace=True)
prices_sectors


# Assume one stock is priced at 5 dollars per share and one stock is priced at 10 dollars per share but their market cap is equal. If we would just add all stock prices, the stock with less outstanding shares and a higher share price is weighted more in the calculation. Therefore we scale all stock prices to 100 for their first year in the S&P 500 to get a good feeling on how the S&P 500 developed over the last decade.

# In[11]:


ser = prices_sectors.groupby("Symbol").first()["Adj Close"]
prices_sectors["temp"] = prices_sectors["Symbol"].map(lambda abb: ser[abb])
prices_sectors["Adj Close Norm"]=100*prices_sectors["Adj Close"]/prices_sectors["temp"]
prices_sectors.drop(["temp"],axis=1,inplace=True)
prices_sectors


# In[12]:


f = lambda x: sum(prices_sectors["Adj Close Norm"][pd.to_numeric(prices_sectors["Year"].str[:4])==x])


# In[13]:


g = lambda x: len(prices_sectors["Adj Close Norm"][pd.to_numeric(prices_sectors["Year"].str[:4])==x])


# In[14]:


overall = [f(x)/g(x) for x in range(2010,2023)]
overall


# In[15]:


sp500 = pd.DataFrame()
sp500["Year"]=range(2010,2023)
sp500["Score"]=overall
sp500


# Using a seaborn plot, we see that the overall trend was very positive. But some market movements are not included in the chart because only yearly data is used (e.g. a bear market in early 2020 because of the Covid 19 pandemic). However the goal of the chart was to get an overall feeling.

# In[16]:


sns.lineplot(data=sp500, x="Year", y="Score", color="red")


# ## Merging our datasets

# Now we want to merge our two datasets multiples_true and prices_sectors. We us the pivot method to do so. Also we get rid of the Adj Close Norm column and just use the adjusted closing prices. [Reference 1](https://jakevdp.github.io/PythonDataScienceHandbook/03.07-merge-and-join.html) [Reference 2](https://pandas.pydata.org/docs/user_guide/reshaping.html)

# In[17]:


mr = prices_sectors.copy()
realyear=[i[:4] for i in mr["Year"]]
mr["Real Year"]=realyear
mr.head()


# In[18]:


mr=mr.pivot(index="Symbol",
         columns="Real Year",
         values= "Adj Close")


# In[19]:


mr.head()


# In[20]:


df = pd.merge(multiples_true, mr, on='Symbol')
df = df.sort_values(by=['Symbol'])
df.drop(["Price","52 Week Low", "52 Week High"],axis=1, inplace=True)
df.head()


# ## Classifying stocks using the K-Means method

# The first thing we want to do is, to see if we can predict, whether the sector of a stock is Real Estate, Industrials or Utilities, using the Price/Sales and the Dividend Yield multiple with the K-Means method. [Reference](https://stackoverflow.com/questions/21415661/logical-operators-for-boolean-indexing-in-pandas)

# First let us have a look on how our data looks:

# In[21]:


df2= df[df['Sector'].eq("Real Estate") | df['Sector'].eq("Utilities") | df['Sector'].eq("Industrials")]


# In[22]:


brush = alt.selection_interval(encodings=["x","y"])

i1 = alt.Chart(df2).mark_circle().encode(
    x="Price/Sales",
    y="Dividend Yield",
    color=alt.condition(brush, "Sector:N", alt.value("green"))
).add_selection(brush) 

i2 = alt.Chart(df2).mark_bar().encode(
    x="Sector:N",
    y=alt.Y("count()", scale=alt.Scale(domain=[0,55])),
    color="Sector:N"
).transform_filter(brush)

alt.hconcat(i1,i2)


# When looking at the data and creating a brush, we clearly see, that there is at least some sort of characteristic properties for stocks from each sector. Industrials tend to have a low dividend yield and a low Price/Sales multiple, Utilities also have a low Price/Sales multiple, but their dividend yield is higher on average and Real Estate stocks have a high Price/Sales multiple. We want to use different machine learning techniques to see, how well these methods can classify given stocks.

# First we use the K-Means method. We don't initialize centroids.

# In[23]:


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, max_iter=10, n_init=5, random_state=2)
kmeans.fit(df2[["Dividend Yield","Price/Sales"]])
kmeans.predict(df2[["Dividend Yield","Price/Sales"]])


# In[24]:


cluster = kmeans.predict(df2[["Dividend Yield","Price/Sales"]])


# Now we want to test whether our results from the prediction using K-Means are actually correct. We use a dictionary and list comprehension to do so. [Reference](https://stackoverflow.com/questions/63697847/changing-label-names-of-kmean-clusters)

# In[25]:


mapping = {0:'Industrials', 1:'Real Estate', 2:'Utilities'}
df2["temp"] = [mapping[i] for i in cluster]
df2["prediction"]=df2["Sector"]==df2["temp"]


# Let's do a chart that shows how well the K-Means algorithm performed.

# In[26]:


c2 = alt.Chart(df2).mark_circle().encode(
    x="Price/Sales",
    y="Dividend Yield",
    color="prediction:N"
)
c2


# We see, the algorithm worked quite well. To evaluate how good it worked, we can do the following calculation:

# In[27]:


len(df2[df2["prediction"]])/len(df2)


# The algorithm predicted 86.5% of the stocks correctly. Let's see if a decision tree works better and how we can avoid overfitting.

# ## Classifying stocks using decision trees

# In[28]:


from sklearn.model_selection import train_test_split
X = df2[["Price/Sales","Dividend Yield"]]
y = df2["Sector"]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9, random_state=0)


# In[29]:


from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(max_leaf_nodes=5)
clf.fit(X_train,y_train)


# In[30]:


clf.score(X_train,y_train)


# In[31]:


clf.score(X_test, y_test)


# Using max_leaf_nodes=5, the performance on our test set is much worse than on the training set (more than 10 percentage points difference) and also worse than with the K_Means method. To see, which max_leaf_nodes value between 2 and 10 gives us the best score on the test set we use a for-loop. [Reference](https://www.kaggle.com/questions-and-answers/169669)

# In[32]:


mlf_candidates = range(2,10)
scores = dict()
train_scores=dict()
for i in mlf_candidates:
    clf = DecisionTreeClassifier(max_leaf_nodes=i)
    clf.fit(X_train,y_train)
    scores[i] = clf.score(X_test,y_test)
    train_scores[i] = clf.score(X_train,y_train)
print(scores)
print(train_scores)


# In[33]:


pd.Series(scores).idxmax()


# In[34]:


scores[7]


# The best possibility for max_leaf_nodes seems to be 7. Let's check overfitting:

# In[35]:


clf7 = DecisionTreeClassifier(max_leaf_nodes=5)
clf7.fit(X_train,y_train)
clf.score(X_train,y_train)


# We see that it is now overfitting way less than before. But there is still a difference of more than 6 percentage points.

# Now let's plot the decision tree with max_leaf_nodes=7:

# In[36]:


import matplotlib.pyplot as plt
from sklearn.tree import plot_tree


# In[37]:


fig = plt.figure(figsize=(20,10))
_ = plot_tree(clf7, 
                   feature_names=clf7.feature_names_in_,
                   class_names=clf7.classes_,
                   filled=True)


# Next let's make a chart of the results of our decision tree, which we will do by coloring the different decison areas in different colors using the DecisionBoundaryDisplay method. [Reference](https://scikit-learn.org/stable/auto_examples/tree/plot_iris_dtc.html)

# In order to get a proper coloring for our plot we need to use a similar trick with a dictionary before and afterwards we see the stocks and decision areas. I did not know how to get a legend for the colors of the scatterplot, because of the trick I used, but white stands for Industrials, black stands for Utilites and grey for Real Estate.

# In[38]:


from sklearn.inspection import DecisionBoundaryDisplay 
import plotly.express as px


# In[39]:


mapp = {'Industrials':0, 'Real Estate':1, 'Utilities':2}
df2["temp2"] = [mapp[i] for i in df2["Sector"]]

x=df2["Price/Sales"]
y=df2["Dividend Yield"]
z=df2["temp2"]


# In[40]:


DecisionBoundaryDisplay.from_estimator(
        clf7,
        X,
        cmap='Pastel1'
    )
plt.scatter(x,y,c=z,cmap='binary')


# ## Linear Regression to predict the Price/Earnings multiple
# The second central project is to look how the Price/Earnings multiple is influenced by current movements in the stock market.

# First we add a new column to our dataframe which gives a multiple for the development of every stock over the last five years.

# In[41]:


df["fy_move"]=df['2022']/df['2017']


# Now we want to restrict the Price/Earnings multiple to the most common area between zero and fifty, as if we would take all values into account, the impact of the outliers would be to large. This makes sense, as a P/E ratio outside of that range would imply a very unusual economic situation of the company.

# In[42]:


df['P/E']=df['Price/Earnings'][(df['Price/Earnings']>0) & (df['Price/Earnings']<50)]


# By looking at the chart we see that there is a positive correlation between the P/E multiple and the movement over the last five years. This was what I expected, as a positive movement in the last five years increases the price of the stock and if we assume a symmetric distribution around zero for the development of earnings, on average the Price/Earnings multiple will go up.

# In[43]:


alt.Chart(df).mark_circle().encode(
    x=alt.X("fy_move", axis=alt.Axis(format='00%', title='Five Year Movement')),
    y=alt.Y("P/E", axis=alt.Axis(title='Price/Earnings Multiple'))
)


# Now let's check these results by using a linear regression. Note: The results can be only taken into account for usual P/E ratios between zero and fifty.

# In[44]:


from sklearn.linear_model import LinearRegression


# In[45]:


regr = LinearRegression(fit_intercept=True)


# In[46]:


data = {'P/E': df['P/E'],
        'fy_move': df['fy_move']
        }


# In[72]:


df_mini = pd.DataFrame(data).dropna()
df_mini.shape
df_mini.head()


# In[48]:


regr.fit(df_mini[["fy_move"]],df_mini["P/E"]).coef_


# We see that if a stock has increased its price a hundred percentage points more in the last five years, it's P/E multiple is predicted to be 2.26 points higher.

# In[49]:


df_mini['P/E_pred'] = regr.predict(df_mini[['fy_move']])


# In[50]:


c4 = alt.Chart(df_mini).mark_circle().encode(
    x=alt.X("fy_move", axis=alt.Axis(format='00%', title='Five Year Movement')),
    y=alt.Y("P/E", axis=alt.Axis(title='Price/Earnings Multiple'))
)

c5 = alt.Chart(df_mini).mark_line(color="black", size=3).encode(
    x="fy_move",
    y="P/E_pred"
)

c4+c5


# The line graph shows us the predicted values for our P/E multiple based on the performance of a stock in the last five years.

# In[51]:


r_squared1 = regr.score(df_mini[["fy_move"]],df_mini["P/E"])
f'The R^2 value of our regression is {r_squared1}'


# Only around 7 percent of the P/E multiple is explained by the five year movement. So we try to add another variable. But to let it make sense, we now predict the five year movement based on two other multiples.

# ## Multivariate Regression to predict the Five Year Movement
# We also calculate the Market Cap/EBITDA multiple. To make that multiple financially correct, we should use Enterprise Value instead of Market Cap (Equity Value), but as these values can't be derived from our data we use this multiple instead and restrict it to the usual values between zero and thirty.

# In[75]:


df_mini["M"]=df["Market Cap"]/df["EBITDA"]


# In[76]:


df_mini["M"]=df_mini["M"][(df_mini["M"]>0)&(df_mini["M"]<30)]
df_mini=df_mini.dropna()


# We want to do a multivariable regression using the P/E multiple and the Market Cap/EBITDA multiple and do a three dimensional plot of the results.

# In[54]:


reg = LinearRegression(fit_intercept=True)


# In[55]:


reg.fit(df_mini[["M","P/E"]],df_mini["fy_move"])


# In[78]:


df_mini['fy_pred'] = reg.predict(df_mini[["M","P/E"]])


# In[57]:


r_squared = reg.score(df_mini[["M","P/E"]],df_mini["fy_move"])
f'The R^2 value of our regression is {r_squared}'


# The R^2 value shows us that only 13% of the the five year movement can be explained by our two multiple. So in general their doesn't seem to be a big linear correlation between our chosen variables. But at least our model is now twice as good as our first linear regression.

# We try this using Matplotlib to plot the stocks and the prediction surface for the five year movement based on the Market Cap/EBITDA and the Price/Earnings multiples and our result is correct, but as we need to plot our graph in three dimensions, it is impossible to actually see the relevant information from this plot. [Reference 1](https://matplotlib.org/stable/gallery/mplot3d/scatter3d.html) [Reference 2](https://stackoverflow.com/questions/57481995/how-to-generate-a-3d-triangle-surface-trisurf-plot-in-python-with-matplotlib)

# In[79]:


fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(df_mini['M'],df_mini['P/E'],df_mini['fy_move'])
ax.set_xlabel('Market Cap/EBITDA')
ax.set_ylabel('Price/Earnings')
ax.set_zlabel('Five Year Movement')
ax.plot_trisurf(df_mini['M'],df_mini['P/E'],df_mini['fy_pred'])


# Therefore I decided to use the plotly.grap_objects library which enables to do really nice and informative three-dimensional interactive charts. [Reference](https://stackoverflow.com/questions/69625661/create-a-3d-surface-plot-in-plotly)

# In[59]:


from scipy.interpolate import griddata
import plotly.graph_objects as go


x = np.array(df_mini['M'])
y = np.array(df_mini['P/E'])
z = np.array(df_mini['fy_pred'])
a = np.array(df_mini['fy_move'])


xi = np.linspace(x.min(), x.max(), 100)
yi = np.linspace(y.min(), y.max(), 100)

X,Y = np.meshgrid(xi,yi)

Z = griddata((x,y),z,(X,Y), method='cubic')
A = griddata((x,y),a,(X,Y), method='cubic')

layout = go.Layout(
    margin=dict(l=80, r=80, t=100, b=80),scene= dict(
        xaxis_title='Market Cap/EBITDA',
        yaxis_title='Price/Earnings',
        zaxis_title='Five Year Movement'
    ))


fig = go.Figure(go.Surface(x=xi,y=yi,z=Z), layout=layout)
fig.add_scatter3d(x=X.flatten(), y=Y.flatten(), z = A. flatten(), mode='markers', 
                  marker=dict(size=2, color=A.flatten(),               
                              colorscale='darkmint'))
fig.show()


# The color of the surface corresponds to the predicted five year movement and the legend for the colors is on the right side of the plot. Every point corresponds to a stock and shows the empirical five year movement of the respective stock. A darker green of a point symbolizes a higher five year movement of the corresponding stock. When going with the mouse over points you also see the values for the different axis.

# From the surface we see, that the predicted five year movement is going up, when both multiples have higher values (positive correlation). 

# ## Stock Price Prediction
# The last thing we wanna do in this project is to use a RandomForestRegressor to predict the 2022 stock prices of Utilities companies based on the five years before (2017,2022). [Reference](https://www.youtube.com/watch?v=BJ6kyj-st9k)

# First we create a new dataframe consisting of the necessary data and make numpy arrays to use our RandomForestRegressor:

# In[60]:


dfu = df[df['Sector']=="Utilities"]


# In[61]:


from sklearn.ensemble import RandomForestRegressor


# In[62]:


x1,x2,x3,x4,x5,y=dfu["2017"],dfu["2018"],dfu["2019"],dfu["2020"],dfu["2021"],dfu["2022"]


# In[63]:


x1,x2,x3,x4,x5=np.array(x1).reshape(-1,1),np.array(x2).reshape(-1,1),np.array(x3).reshape(-1,1),np.array(x4).reshape(-1,1),np.array(x5).reshape(-1,1)


# In[64]:


x=np.concatenate((x1,x2,x3,x4,x5),axis=1)


# We instantiate such a regressor, split our data and then fit the training data and predict the test data.

# In[65]:


model=RandomForestRegressor(n_estimators=100, max_features=5, random_state=0)


# In[66]:


X1_train,X1_test,y1_train,y1_test = train_test_split(x, y, train_size=0.9, random_state=0)


# In[67]:


model.fit(X1_train,y1_train)


# In[68]:


prediction = model.predict(X1_test)


# In order to get a good plot, we need to get a dataframe, where we have columns with the actual data of the stocks in the testset and the predicted data in other columns.

# In[69]:


df4=pd.DataFrame(X1_test)
df5=pd.DataFrame(prediction)
df6=pd.concat([df4,df5],axis=1)
df7=pd.DataFrame(y1_test.values)
df8=pd.concat([df4,df7],axis=1)


# In[70]:


df_plot=pd.concat([df6,df8],axis=0).transpose()
df_plot["Year"]=range(2017,2023)
df_plot.columns = ["Prediction1","Prediction2","Prediction3","Actual1","Actual2","Actual3","Year"]
df_plot


# We see that there are three stocks in our testset. We want to plot the actual stock prices for these three stocks in red and the predicted stock data in blue. [Reference](https://github.com/altair-viz/altair/issues/968)

# In[71]:


base = alt.Chart(df_plot.reset_index()).encode(x=alt.X('Year',scale=alt.Scale(domain=(2017,2022)), axis=alt.Axis(format='', title='Year')))
alt.layer(
    base.mark_line(color='blue').encode(y=alt.Y('Prediction1',axis=alt.Axis(title='Price'))),
    base.mark_line(color='red').encode(y='Actual1'),
    base.mark_line(color='blue').encode(y='Prediction2'),
    base.mark_line(color='red').encode(y='Actual2'),
    base.mark_line(color='blue').encode(y='Prediction3'),
    base.mark_line(color='red').encode(y='Actual3')
).interactive()


# As we can see, the accuracy of RandomForestRegressor was quite different. The best result is achieved for the second stock, where you have to zoom in to see a difference between the red and the blue line. But I would say the overall result is quite good. Whether a stock went up or down was predicted right for all three stocks, eventhough two of them dropped in price the year before.

# ## Summary

# To summarize we can say that both the K-Means algorithm and Decision Trees work quite well when it comes to classifying stocks by their sector. Using the optimal number for max_leaf_nodes, we get a slightly better result using Decision trees, but the difference is only four percentage points.
# We also saw a small correlation between the past movement of a stock and some of the key financial metrics. Actually I expected an even bigger correaltion and was quite surprised by the low R^2 scores of the regressions.
# When using a RandomForestRegressor to analyze stock price movements, we can get at least an idea into which direction the stock price will move next. How well our time series analysis predicted future stock prices was very different for different stocks.

# ## References
# 
# Your code above should include references.  Here is some additional space for references.

# * What is the source of your dataset(s)?

# https://github.com/datasets/s-and-p-500-companies-financials/blob/master/data/constituents-financials.csv
# https://www.kaggle.com/datasets/andrewmvd/sp-500-stocks/code

# Full list of references including further sites helping with general ideas:

# https://jakevdp.github.io/PythonDataScienceHandbook/03.07-merge-and-join.html
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.concat.html
# https://stackoverflow.com/questions/21415661/logical-operators-for-boolean-indexing-in-pandas
# https://www.kaggle.com/questions-and-answers/169669
# https://www.kaggle.com/code/kallefischer/sp-500-prediction-67-5-accuracy
# https://stackoverflow.com/questions/57207108/typeerror-ufunc-isfinite-not-supported-for-the-input-types-and-the-inputs-co
# https://stackoverflow.com/questions/57481995/how-to-generate-a-3d-triangle-surface-trisurf-plot-in-python-with-matplotlib
# https://stackoverflow.com/questions/63697847/changing-label-names-of-kmean-clusters
# https://www.youtube.com/watch?v=BJ6kyj-st9k
# https://stackoverflow.com/questions/69625661/create-a-3d-surface-plot-in-plotly
# https://pandas.pydata.org/docs/user_guide/reshaping.html
# https://matplotlib.org/stable/tutorials/colors/colormaps.html
# https://scikit-learn.org/stable/auto_examples/tree/plot_iris_dtc.html
# https://github.com/altair-viz/altair/issues/968
# 

# ## Submission
# 
# Using the Share button at the top right, **enable Comment privileges** for anyone with a link to the project. Then submit that link on Canvas.

# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=5b7c89e2-b326-479c-b0b9-d9986e29105a' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>
