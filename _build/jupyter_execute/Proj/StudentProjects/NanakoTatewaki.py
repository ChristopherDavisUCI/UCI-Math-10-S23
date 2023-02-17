#!/usr/bin/env python
# coding: utf-8

# # Prediction of Canadian Car Accident
# 
# Author: Nanako Tatewaki
# 
# 59793326
# 
# Course Project, UC Irvine, Math 10, F22

# ## Introduction
# 
# I would like to use the "Canadian Car Accidents 1994-2014" dataset from the Kaggle (which we haven't utilize this dataset for this class) to predict the data of the accident (gender, weather condition, road condition,etc...). In this project, I will utilize some method that we covered from the lecture, such as scikit-learn, Altair chart, classifier, and more for the machine learning. And for the extra topic, I choose the K Nearest Neighbor Classifier and Logistic Regression.

# ## Importing Files
# 

# ### Dataset from the Kaggle

# import libraries and the dataset

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import altair as alt
import pandas as pd
import zipfile
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss
import plotly.express as px
from sklearn.tree import DecisionTreeClassifier
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans


# In[2]:


df = pd.read_csv("caraccidents.zip",compression='zip')


# ### Clean Data

# Since the Data is too huge, we will choose 100,000 data randomly and clean the data to utilize later on. 

# 

# In[3]:


df.shape


# In[ ]:


df_sam = df.sample(100000, random_state = 1).copy()
#drop some unnecessary columns
df_sam = df_sam.drop(columns=['V_ID', 'P_ID'])
df_sam


# Now, we will explore the interaction of gender and age.  First, we will figure out what kind of genders are in the column of "P_SEX",  and what ages are in the column of "P_AGE" by using the value counts method.

# Most of the data seems numeric, however, most of the columns are categorical value explained in the original PDF source.

# The data frame contains missing values and uneven object types .

# In[5]:


#U = ,N =
df_sam["P_SEX"].value_counts()


# In[6]:


#contains '12' and 12
df_sam["C_MNTH"].unique()


# 

# Convert all the columns except for gender to numeric object.

# In[7]:


cols = [c for c in df_sam.columns if c != "P_SEX"]


# Remove the entries which couldn't be converted = missing value

# In[8]:


df_sam[cols] = df_sam[cols].apply(pd.to_numeric, errors='coerce')
df_sam = df_sam.dropna()
#convert to int object
df_sam[cols] = df_sam[cols].astype(int)


# In order to clarify the data for the gender, we will drop U (unknown) and N (not applicable).

# In[ ]:


df_sam = df_sam[(df_sam["P_SEX"]=="M") | (df_sam["P_SEX"]=="F")]
#more descriptive
df_sam = df_sam.replace({"F": "Female", "M": "Male"}).copy()


# Somehow, this data lacks of datetime information which is important for the data analyzing so we would do our best to accommodate.

# In[10]:


df_sam['date'] = pd.DatetimeIndex(df_sam['C_YEAR'].map(str) + '-' + df_sam['C_MNTH'].map(str))
df_sam = df_sam.set_index('date').sort_index().copy()


# Now the data is ready to be investigated.

# In[11]:


df_sam


# ## Investigation

# ### Gender, Age, Fatality

# Since our data is larger than the default limit of Altair, will increase by 100,000 (not sure if necessary).

# In[12]:


alt.data_transformers.enable('default', max_rows=100000)


# Following is a histogram showing the number of accidents by age (bin = 10)

# df_sam is still big data that we got error, so we have to get another sample, but it is adequate to portrait the entire trend.

# In[13]:


dfs_alt = df_sam.sample(10000, random_state = 1)
alt.Chart(dfs_alt).mark_bar().encode(
    x= alt.X("P_AGE",bin = True),
    y="count()",
    color = "P_ISEV:N",
).properties(
    height=350,
    width=350
).facet(
    column="P_SEX"
)


# From this, you can notice some points, such that, 20-30 years old has the most number of the accidents and Men has more number of the accident than the women has. In addition, women has higher ratio of the injury than the men. However, men has higher rate of fatality than the women. 

# ### Accident Reports vs Day and Hour of the day

# Here are three charts to visualize the accident density by the hour and day of week. First Altair chart is to visualize the number of records for each day and hour of the week. Second Altair chart is to see the data of severity with x-axis of number of records and y-axis of each hours. The third chart is to visualize the severity ratio by days of week and the number of records. 

# In[14]:


right = alt.Chart(dfs_alt).mark_bar().encode(
    y= "C_HOUR:N",
    x="count()",
    color = alt.Color("P_ISEV:N",scale=alt.Scale(scheme="redpurple" )),
).properties(
    height = 400
)

c_count = alt.Chart(dfs_alt).mark_rect().encode(
    x= "C_WDAY:N",
    y="C_HOUR:N",
    color = alt.Color('count()',scale=alt.Scale(scheme="redpurple" ))
)

c_text = alt.Chart(dfs_alt).mark_text(color="white").encode(
    x="C_WDAY:N",
    y="C_HOUR:N",
    text="count()"
)

center = (c_count+c_text).properties(
    height=400,
    width=300
)

h = alt.hconcat(center,right)

bottom = alt.Chart(dfs_alt).mark_bar().encode(
    x= "C_WDAY:N",
    y="count()",
     color = alt.Color("P_ISEV:N",scale=alt.Scale(scheme="redpurple" )),
).properties(
    width = 300
)

alt.vconcat(h,bottom)


# From these charts, we can see that the around 3-5pm has the peak of number of the accidents each day and  Friday 5pm has the most number of the accidents from the week. 

# ## Machine Learning

# Again, this data frame is mostly categorical values and is challenging to apply machine learning to extract meaningful insights

# Severity of the accident (p-isev 1 = no injury 2 = injury 3 = fatal) could be an candidate for a target value predicted by other quantities.

# ### Decision Tree Classifier

# We will apply a simple Decision Tree Classifier model to see if there is any correlation

# First, convert gender category to the discrete value. 

# In[15]:


df_sam["IsFemale"] = df_sam["P_SEX"].map(lambda x: 1 if x == 'Female' else 0)


# In[16]:


dclf = DecisionTreeClassifier()
features = [i for i in df_sam.columns if pd.api.types.is_numeric_dtype(df_sam[i]) and i != 'P_ISEV']
X = df_sam[features]
y = df_sam['P_ISEV']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
dclf.fit(X_train, y_train)
dclf.score(X_test, y_test)


# We should not be executing the learning without the treatments of categorical values. But, our point here is to show that the accuracy of our model seems very low

# You can see below that the level of severity overlaps almost completely and hard to classify.

# In[17]:


alt.Chart(dfs_alt).mark_circle().encode(
    x="P_AGE",
    y=alt.Y("V_YEAR", scale=alt.Scale(zero=False)),
    color = alt.Color("P_ISEV:N",scale=alt.Scale(scheme="redpurple" ))
)


# It does only slightly better than this randomly-trained model

# In[18]:


y_shuffle = y_train.sample(frac = 1)
dclf.fit(X_train, y_shuffle)
dclf.score(X_test, y_test)


# However, we could possibly argue that some of the features actually helped the model to predict the extra percent accuracy by using the feature importance attribute. 

# In[19]:


pd.Series(dclf.feature_importances_, index=dclf.feature_names_in_).sort_values(ascending=False)


# Lets use the top 3 features and apply onehotencoder so that the categorical values can be processed.

# In[20]:


encoder = OneHotEncoder()
encoder.fit(df_sam[["C_WDAY", "C_MNTH",'C_CONF']])
df_dec = df_sam.copy()
df_dec[list(encoder.get_feature_names_out())] = encoder.transform(df_sam[["C_WDAY", "C_MNTH",'C_CONF']]).toarray()


# Other columns are still included but their importance is so low that we can ignore these.

# In[23]:


features = [i for i in df_dec.columns if pd.api.types.is_numeric_dtype(df_dec[i]) and i != 'P_ISEV']
X = df_dec[features]
y = df_dec['P_ISEV']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
dclf.fit(X_train, y_train)
dclf.score(X_test, y_test)


# The outcome for this onehotencoder seems unchanged compare to the decision tree classifier, therefore we can conclude that it is hard to predict the severity of the accidents by decision tree classifier. 

# ### K Nearest Neighbor Classifier

# Let us again perform a simple classifier model, however this time we use knn classifier with smaller features.

# In[25]:


kclf = KNeighborsClassifier(n_neighbors=10)
features = ['P_AGE','V_YEAR','C_YEAR','C_HOUR']
X = df_sam[features]
y = df_sam['P_ISEV']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
kclf.fit(X_train, y_train)
kclf.score(X_test, y_test)


# The result doesn't seem very good either, but not sure if our choice of neighbors was optimal.

# Below we compared the test and train scores of our model for each k-neighbors.

# In[ ]:


df_scores = pd.DataFrame({"k":range(1,100),"train_score":np.nan,"test_score":np.nan})
#original df_sam takes too much time to run
df_ss = df_sam.sample(1000,random_state=1).copy()
X = df_ss[features]
y = df_ss['P_ISEV']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
kclf.fit(X_train, y_train)
for i in df_scores.index:
    kclf = KNeighborsClassifier(n_neighbors=i+1)
    kclf.fit(X_train, y_train)
    df_scores.loc[i,["train_score","test_score"]] = [kclf.score(X_train, y_train),kclf.score(X_test, y_test)]

df_scores["kinv"] = 1/df_scores.k
ctrain = alt.Chart(df_scores).mark_line(color = 'black').encode(
    x = "kinv",
    y = "train_score"
)
ctest = alt.Chart(df_scores).mark_line(color="#FF007F").encode(
    x = "kinv",
    y = "test_score"
)

ctrain+ctest


# We see that the while train score improves the test score remains flat, so around k = 20 is an optimal choice, yet accuracy is still 55%, in which it is hard to predicts the correction of severity.

# Let us investigate other target values 

# In[28]:


kclf = KNeighborsClassifier(n_neighbors = 20)
X = df_sam[features]
y1 = df_sam['V_TYPE']
X_train, X_test, y1_train, y1_test = train_test_split(X, y1, train_size=0.8)
kclf.fit(X_train, y1_train)
kclf.score(X_test, y1_test)


# the knn classifier predicts the vehicle type of accidents by 90%.  but is this any better prediction than the severity?

# If we look at the occurrence of accidents by the vehicle type by utilizing the value_count method, we see that 1(light duty vehicle) is around 90% of the entire accident cases, so any other random prediction will likely score the similar accuracy.

# In[29]:


df_sam["V_TYPE"].value_counts()


# Our conclusion for the classifier part is that we were unable to extract strong correlation predicting the severity of the accident and most other categorical values as well. 

# ### Linear and Polynomial Regressor

# What we could do instead is predicting the occurrences of accidents given some conditions.

# Here is the simplified data we will use for our regression model.

# In[30]:


df_lin = pd.DataFrame(df_sam.index.value_counts()).sort_index().rename(columns={"date": "count"}).reset_index().copy()
df_lin['month'] = df_lin['index'].dt.month
df_lin['year'] = df_lin['index'].dt.year
df_lin


# In[31]:


alt.Chart(df_lin).mark_line(color = 'grey').encode(
    x="index",
    y=alt.Y('count:Q',
        scale=alt.Scale(zero=False)),
).properties(
    height=400,
    width=700
)


# From this chart, we can see a clear trend that each year has the similar cycle by month and average of the cases is decreasing annually. 

# Let us visualize the trend in the following way

# In[32]:


trend = alt.Chart(df_lin).mark_line().encode(
    x="month:O",
    y=alt.Y('count:Q',
        scale=alt.Scale(zero=False)),
         color = alt.Color("year:O",scale=alt.Scale(scheme="greys" ))
).properties(
    height=400,
    width=700
)
#average trend
mean = alt.Chart(df_lin).mark_line(color='#FF007F').encode(
    x='month:O',
    y='mean(count)'
)
trend+mean


# We see that the the number of accidents are at lowest at April and peaks around fall-winter annually.

# 

# The cyclic pattern should be better fitted by polynomial regression of some degree rather than a linear graph

# We will visualize the loss for each of d-degree polynomial regression.

# In[33]:


from sklearn.metrics import mean_squared_error
train_dict={}
test_dict={}

X_train, X_test, y_train, y_test = train_test_split(
    df_lin[['year',"month"]],
    df_lin['count'],
    train_size=0.8,
    random_state=1
)
for n in range(1,20):
    reg = LinearRegression()
    
    for i in range (2,n+1):
        X_train['m'+str(i)] = X_train['month']**i
        X_test['m'+str(i)] = X_test['month']**i
        
    reg.fit(X_train, y_train)
    train_error = mean_squared_error(y_train, reg.predict(X_train))
    train_dict[n] = train_error
    test_error = mean_squared_error(y_test, reg.predict(X_test))
    test_dict[n] = test_error


# In[34]:


#creating a dataframe for altair chart
train_ser = pd.Series(train_dict)
test_ser = pd.Series(test_dict)
train_ser.name = "train"
test_ser.name = "test"
df_loss = pd.concat((train_ser, test_ser), axis=1)
df_loss.reset_index(inplace=True)
df_loss.rename({"index": "poly_degree"}, axis=1, inplace=True)
df_melted = df_loss.melt(id_vars="poly_degree", var_name="Type", value_name="Loss")


# In[35]:


alt.Chart(df_melted).mark_line().encode(
    x="poly_degree",
    y=alt.Y('Loss',
        scale=alt.Scale(zero=False)),
    color = alt.Color("Type",scale=alt.Scale(scheme="redpurple" ))
)


# we see that the loss and the difference of losses are minimum  around d=4-14, so we choose d = 4 for efficiency.

# In[38]:


cols = ['year']
for i in range (1,5):
        df_lin['m'+str(i)] = df_lin['month']**i
        cols.append('m'+str(i))


# In[39]:


reg = LinearRegression()
reg.fit(df_lin[cols],df_lin['count'])
df_lin["Pred"] = reg.predict(df_lin[cols])


# In[40]:


base = alt.Chart(df_lin).mark_line(color = 'grey').encode(
    x="index:T",
    y=alt.Y('count:Q',
        scale=alt.Scale(zero=False)),
)

pred = alt.Chart(df_lin).mark_line(color = '#FF007F').encode(
    x="index:T",
    y=alt.Y('Pred:Q',
        scale=alt.Scale(zero=False)),
)

(base+pred).properties(
    height=400,
    width=700
)


# Here we can see that our polynomial model predicts the count value accurately.

# In[41]:


pd.Series(reg.coef_,reg.feature_names_in_)


# If we look at the coefficients we see that m4 is already very small and not requiring higher degree coefficients.

# ### K-Means Clustering

# now we understood the overall trends of the car accidents in macroscopic. 

# we would like to investigate for smaller scope of view, however this data does not provide the date of an accident.

# here we extracted all incidents occured in 2014.

# In[ ]:


df_y = df[df['C_YEAR']==2014].copy()
cols = [c for c in df_y.columns if c != "P_SEX"]
df_y[cols] = df_y[cols].apply(pd.to_numeric, errors='coerce')
df_y = df_y.dropna()
df_y[cols] = df_y[cols].astype(int)
df_y = df_y[(df_y["P_SEX"]=="M") | (df_y["P_SEX"]=="F")]
df_y


# We take a similar approaches as the beginning to clean the data

# To analyze the data by timescale, we used k-means clustering to artificially assign a date from 1-28 to each incidents. The clustering is done using the day of week and the weather condition, because it makes sense that the incidents with equal day of the week and the weather condition to have a high chance of them being the same day.

# In[51]:


kmeans = KMeans(n_clusters = 28)
dcol = ['C_WDAY','C_WTHR']
kmeans.fit(df_y[dcol])
df_y["cluster"] = kmeans.predict(df_y[dcol])+1
dfy_alt = df_y.sample(20000, random_state = 1)
alt.Chart(dfy_alt).mark_circle(size = 300).encode(
    x = "C_WDAY:N",
    y = "C_WTHR:O",
    color = "cluster:N"
).properties(
    width=400,
    height = 400
)


# Now we can investigate the frequency pattern with our artificial date-time values.

# In[52]:


df_y['date'] = pd.DatetimeIndex(df_y['C_YEAR'].map(str) + '-' + df_y['C_MNTH'].map(str) + '-' + df_y['cluster'].map(str)+ ' ' + df_y['C_HOUR'].map(str)+':00')
df_y = df_y.set_index('date').sort_index().copy()
df_y['C_WDAY'] = df_y['C_WDAY'].replace({1: "Monday", 2: "Tuesday",3: "Wednesday", 4: "Thursday",5: "Friday", 6: "Saturday",7: "Sunday"}).copy()
df_y['C_WTHR'] = df_y['C_WTHR'].replace({1: "Sunny", 2: "Cloudy",3: "Raining", 4: "Snowing",5: "Hail", 6: "Fog", 7: "Windy"}).copy()
df_y


# Though we shouldn't be looking at individual values too closely since it's artificial, we can definitely see the trends by the day of  week, hour, and monthly.

# In[ ]:


sel = alt.selection_multi(fields=["C_WDAY"], empty="none")
base = alt.Chart(df_y.sample(20000).reset_index()).mark_line().encode(
    x="date:T",
    y=alt.Y('count():Q',
        scale=alt.Scale(zero=False)),
        tooltip=["date", "count()", "C_WDAY", "C_WTHR"],
        opacity=alt.condition(sel, alt.value(1.5), alt.value(0.2)),
        color = 'C_WDAY:N'
).properties(
    height=400,
    width=700
).interactive().add_selection(sel)

text = alt.Chart(df_y.sample(20000)).mark_text(y=20, size=20).encode(
    text="C_WDAY",
    opacity=alt.condition(sel, alt.value(1), alt.value(0))
)

c = base+text
c




# If we look closely to the interactive chart above, we can see that the clustering separated the day of the week fairly accurately, where each color is repeating frequencies by 7 days. In addition, we can see the drop in April and peak in autumn, peak on Friday and at 5pm in which, we saw earlier in this project.

# We used the weather condition to cluster the dates, but it turned out that the weather condition has strong correlation with the number of incidents.

# In[ ]:


sel2 = alt.selection_multi(fields=["C_WTHR"], empty="none")
base = alt.Chart(dfy.sample(20000).reset_index()).mark_line().encode(
    x="date:T",
    y=alt.Y('count():Q',
        scale=alt.Scale(zero=False)),
        opacity=alt.condition(sel2, alt.value(1), alt.value(0.5)),
        color = alt.Color('C_WTHR:N', scale = alt.Scale(scheme = 'Category10'))
).properties(
    height=400,
    width=700
).interactive().add_selection(sel2)

text = alt.Chart(dfy.sample(20000)).mark_text(y=20, size=20).encode(
    text="C_WTHR",
    opacity=alt.condition(sel2, alt.value(1), alt.value(0))
)

c = base+text
c


# It is obvious that the sunny is most occurred and evenly spread out, and we see that the snowing occurs in winter and has correlation with high count of accidents. We also see that for fog condition, the phase is tall and narrow which implies high number of accidents on foggy days.

# ### Logistic Regression

# Finally we will use Logistic Regression to see if we can predict weather or not the accident was fatal.

# In[53]:


df_log = df_sam.copy()
df_log['Is_Fatal'] = df_log["P_ISEV"].map(lambda x: 1 if x == 3 else 0)
lclf = LogisticRegression()
features = [i for i in df_log.columns if pd.api.types.is_numeric_dtype(df_log[i]) and i != 'P_ISEV' and i != 'Is_Fatal' and i != 'C_SEV']
X = df_log[features]
y = df_log['Is_Fatal']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8)
lclf.fit(X_train, y_train)
lclf.score(X_train, y_train),lclf.score(X_test, y_test)



# At first, it seemed like we finally reached something meaningful in our paper. However, if we look at the percentage of fatal accidents, it is actually less than 1% of the total incidents which tells us that out model does not predict fatality any better than random prediction.

# In[54]:


df_log['Is_Fatal'].value_counts()


# 

# ## Summary
# 
# For this project, we begin with cleaning the data, and then explore it by using the pandas method. For the machine leaning, we choose five topics such as Decision Tree Classifier, KNN Classifier, Linear and Polynomial Regrresor, K_Mean Clustering, and Logistic Regression. From these results, we can conclude that the this data that I got from Kaggle was hard to predict the data since it mostly contains the categorial values. 

# ## References
# 
# Your code above should include references.  Here is some additional space for references.

# * What is the source of your dataset(s)?

# Kaggle

# https://www.kaggle.com/datasets/tbsteal/canadian-car-accidents-19942014?select=NCDB_1999_to_2014.csv

# https://www.kaggle.com/code/lastdruid/collision-descriptive-analysis-and-visualization

# * List any other references that you found helpful.

# from zipfile to dataframe  https://stackoverflow.com/questions/26942476/reading-csv-zipped-files-in-python

# https://www.linkedin.com/pulse/change-data-type-columns-pandas-mohit-sharma#:~:text=1.-,to_numeric(),floating%2Dpoint%20numbers%20as%20appropriate.

# https://www.fullstory.com/blog/categorical-vs-quantitative-data/

# https://stackoverflow.com/questions/21415661/logical-operators-for-boolean-indexing-in-pandas

# Worksheet 10,11,12,14,15 and Lecture notes 

# Color Scheme https://vega.github.io/vega/docs/schemes/

# 

# 

# ## Submission
# 
# Using the Share button at the top right, **enable Comment privileges** for anyone with a link to the project. Then submit that link on Canvas.

# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=cb4932ac-cfc3-417b-b078-f7f7a018b45e' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>
