#!/usr/bin/env python
# coding: utf-8

# # Customer Personality Analysis
# 
# Author: Aner Huang
# 
# Course Project, UC Irvine, Math 10, F22

# ## Introduction
# 
# - <font color = Brown>For this project, I chose "Customer Personality Analysis." It is about the detailed analysis of a company's ideal customers. It helps a business to better understand its customers and makes it easier for them to modify products according to the specific needs, behaviors and concerns of different types of customers.Customer personality analysis helps a business to modify its product based on its target customers from different types of customer segments. Since it contains a lot of factor to analyze, I only focus on the catalog of "people", including "Age", "Income", "Education",and "Marital_status". 

# ## Section 1: Overview and Clean Dataset

# - To begin, I will firstly import some packages that I am going to use in this project and analysis. 
# 
# - Then, I will load my dataset and show some basic information of my dataset. 
# 
# 

# In[46]:


import pandas as pd
import numpy as np


# In[2]:


# Read the dataset
df=pd.read_csv("Costomer_Personality.csv")
df


# In[3]:


# Dimension of dataset
df.shape


# In[4]:


# Counting numbers of missing values in each column
df.isna().sum()


# - We can see that we have 24 missing values in the colume "Income", we can fill these bad datas as the median of the colume of "Income" using `fillna`  

# In[5]:


df['Income']=df['Income'].fillna(df['Income'].median())


# ## Section1.1: A Brief Introduction of Dataset
# - In order for us to better analyze this dataset, I will make a better clear names for those columns that have an ambiguious name and I will also clarity the meaning of each columns for people to understand.
# - For the following, I normalize the dataset by using the method of `rename`.

# In[6]:


# List out all the names of columns
df.columns


# In[7]:


# Normalizing Dataset
df.rename({"Dt_Customer ":"Date","MntWines":"Wines","MntFruits":"Fruits","MntMeatProducts":"Meat","MntFishProducts":"Fish","MntSweetProducts":"Sweet","MntGoldProds":"Gold","NumDealsPurchases":"Deals","NumWebPurchases":"Web","NumCatalogPurchases":"Catalog","NumWebVisitsMonth":"WebVisit"},axis=1,inplace=True)


# - Create another feature "Total" indicating the total amount spent by the customer in various categories over the span of two years.
# - Classify the objects in "Marital_Status" to extract the living situation of couples.
# - Dropping some of the redundant features and other features that I am not going to analyze in this project using `drop`. 

# In[8]:


df["Total"] = df["Wines"]+df["Fruits"]+df["Meat"]+df["Fish"]+df["Sweet"]+df["Gold"]
df['Marital_Status'] = df['Marital_Status'].replace({'Married':'Relationship', 'Together':'Relationship','Divorced':'Alone','Widow':'Alone','YOLO':'Alone', 'Absurd':'Alone'})
to_drop = ["Kidhome","Teenhome", "Z_CostContact", "Z_Revenue"]
df = df.drop(to_drop, axis=1)


# **Brief Introduction of Columns:**
# <font color=brown>
# - People:
# ID: Customer's unique identifier
# Year_Birth: Customer's birth year
# Education: Customer's education level
# Marital_Status: Customer's marital status
# Income: Customer's yearly household income
# Kidhome: Number of children in customer's household
# Teenhome: Number of teenagers in customer's household
# Date: Date of customer's enrollment with the company
# Recency: Number of days since customer's last purchase
# Complain: 1 if the customer complained in the last 2 years, 0 otherwise
# 
# - Products:
# Wines: Amount spent on wine in last 2 years
# Fruits: Amount spent on fruits in last 2 years
# Meat: Amount spent on meat in last 2 years
# Fish: Amount spent on fish in last 2 years
# Sweet: Amount spent on sweets in last 2 years
# Gold: Amount spent on gold in last 2 years
# 
# - Promotion:
# Deals: Number of purchases made with a discount
# AcceptedCmp1: 1 if customer accepted the offer in the 1st campaign, 0 otherwise
# AcceptedCmp2: 1 if customer accepted the offer in the 2nd campaign, 0 otherwise
# AcceptedCmp3: 1 if customer accepted the offer in the 3rd campaign, 0 otherwise
# AcceptedCmp4: 1 if customer accepted the offer in the 4th campaign, 0 otherwise
# AcceptedCmp5: 1 if customer accepted the offer in the 5th campaign, 0 otherwise
# Response: 1 if customer accepted the offer in the last campaign, 0 otherwise
# 
# - Place: 
# Web: Number of purchases made through the company’s website
# Catalog: Number of purchases made using a catalogue
# Store: Number of purchases made directly in stores
# WebVisits: Number of visits to company’s website in the last month

# In[9]:


# Description of Data
df.describe()


# ## Section2: The Relationships between Customer's life status and Total Amount of Purchase

# ### 2.1. Generation

# - For this section, I am interested in analyzing the relationship between Costomer's Age and the total purchases they made. 
# - For total purchases, I will need to make a new column contains the total number they purchased in the last two year by adding up the amount of Wines, Fruits, Meat, Fish, Sweet, and Gold. 
# - I will also create a new column "Age" represents customer's age also a "generation" column represents customer's generation and narrow the range to the age under 80, thus we will also have generation 2-7. 
# - I will also include charts about the distribution of different generation.

# In[10]:


#Current year minus the year of birth will be the age of customers 
df["Age"] = 2022-df["Year_Birth"]
df = df[df["Age"]<80] #Narrow my age range


# - Using `map` method to create a new column called "Generation" to specify different age group. 
# - But first, I would need to make the "Age" column becomes 'str' instead of 'int', so the numbers in the "Age" column does not have any numerical meaning, instead it will represents the age group.

# In[45]:


df["Age"]=df["Age"].apply(str)
df["Generation"] = df["Age"].map(lambda x: x[:1])


# In[12]:


df.groupby("Generation", sort=True).mean()


# In[13]:


df["Generation"].value_counts()


# In[14]:


# Using groupby to find out the distribution of the custormers' generation.
for gp, df_mini in df.groupby("Generation"):
    print(f"The generation is {gp} and the number of rows is {df_mini.shape[0]}.")


# In[15]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[16]:


#Plot of the distribution of generation
plt.figure(figsize=(8,8))
sns.distplot(df["Age"],color = 'turquoise')
plt.show()


# - As we can see from the `groupby` method, generation 4,5,and 6 contains a larger portion of customers. And later, I double check with the plot chart to see its indeed Customer's age around 40-50 goes to the peak.

# **Customer's income**
# - Then, I want to see the relationship between customer's income and the total amount of purchases they made.
# - Before that, I would like to narrow down the range of income, in case that the number is too large to considered as outliers. 

# In[17]:


df = df[df["Income"]<100000]


# In[18]:


import altair as alt
brush = alt.selection_interval()
c1 = alt.Chart(df).mark_circle().encode(
    x=alt.X('Income', scale=alt.Scale(zero=False)),
    y='Total',
    color='Generation:N',
    tooltip=["ID", "Income", "Total"]
).add_selection(brush)

c2= alt.Chart(df).mark_bar().encode(
    x = 'ID',
    y='Total'
).transform_filter(brush)

c1|c2


# <font color=red>Conclusion: We can see from this chart that there might be a positive relationship between the customers' income and their total purchase. Later, I will use. regression to see if there's a relation lie between them.

# ## Linear and Polynomial Regression

# In[19]:


from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(df[["Income"]], df["Total"])
df["Pred"]=reg.predict(df[["Income"]]) 
df.head()


# In[20]:


c = alt.Chart(df).mark_circle().encode(
    x=alt.X('Income', scale=alt.Scale(zero=False)),
    y=alt.Y('Total', scale=alt.Scale(zero=False)),
    color="ID"
)
c1=alt.Chart(df).mark_line(color="red").encode(
    x=alt.X('Income', scale=alt.Scale(zero=False)),
    y="Pred"
)
c+c1


# <font color=red>By the graph above, we can easily confirm that there is a positive trend between customers' income and total amount of purchase. 

# In[21]:


df["I2"]=df["Income"]**2
df["I3"]=df["Income"]**3
poly_cols = ["Income","I2", "I3"]
reg2 = LinearRegression()
reg2.fit(df[poly_cols], df["Total"])
df["poly_pred"] = reg2.predict(df[poly_cols])


# In[22]:


c = alt.Chart(df).mark_circle().encode(
    x=alt.X('Income', scale=alt.Scale(zero=False)),
    y=alt.Y('Total', scale=alt.Scale(zero=False)),
    color="ID"
)

c1 = alt.Chart(df).mark_line(color="black").encode(
    x=alt.X('Income', scale=alt.Scale(zero=False)),
    y="poly_pred"
)

c+c1


# <font color=red>Using polynomial regression to check, we can see the line it's not strictly positive or negative, but its mostly positive. 

# ## Logistic Regression

# - For this section, I want to add one feature in our analysis: "Marital_Status". I am interested in predicting the customer' marital status by their income and total amount spent on products. 

# In[23]:


from sklearn.linear_model import LogisticRegression #import
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np


# In[24]:


# Make a sub-dataframe that only containes the necessary input that we want to predict
cols = ["Income","Total"]
df["Rel1"]=(df["Marital_Status"]== "Relationship") #Make the new colnmn that returns "True" if the customer is in a relationship, otherwise returns "False".


# - Because our original dataset has a large sample size, making a train_test_split to divide dataset would make better and more accurate prediction. 

# In[25]:


X_train, X_test, y_train, y_test = train_test_split(df[cols], df["Rel1"], test_size=0.4, random_state=0)


# In[26]:


clf=LogisticRegression()
clf.fit(X_train, y_train) #fit
(clf.predict(X_test) == y_test).sum() # Frequency 


# In[27]:


# The proportion that we made correct prediction based on the whole dataset.
(clf.predict(X_test) == y_test).sum()/len(X_test)


# - The following step is to make sure that we get the right coefficient by specifing the index.

# In[28]:


clf.coef_
Income_coef,Total_coef=clf.coef_[0]
Total_coef


# - From the coefficient that we get from the logistic prediction, we can tell that there is little relationship between the input:Income and Total with the output:Maritial_Status. Thus, it means it might not be higher income and more total will indicate if a customeris in a relationship. 

# - <font color = brown>Q:What will our model predict if we have the income for 71613 and the total amount is 776?

# In[29]:


sigmoid = lambda x: 1/(1+np.exp(-x))
Income = 71613
Total = 776
sigmoid(Income_coef*Income+Total_coef*Total+clf.intercept_)


# <font color=red> Therefore,our model predicts that this customer with(income is $71613 and total is 776) has a 69.5% chance of being in a relationship. 
# - <font color=black>Then we will double check with predict_proba.

# In[30]:


clf.predict_proba([[Income,Total]]) 


# - The first array says that there is a 30.5% chance of this customer to be single(not in a relationship), and the second array gives the same result as the sigmoid function gives.

# ## K-nearest Neighbor Classification

# In[31]:


#Import
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import log_loss


# In[32]:


clf = KNeighborsClassifier()
clf.fit(X_train, y_train)
loss_train=log_loss(y_train, clf.predict_proba(X_train))
loss_test=log_loss(y_test,clf.predict_proba(X_test))


# In[33]:


loss_train


# The log loss for x_train and y_train is about 0.501. 

# In[34]:


loss_test


# The log loss for x_test and y_test is about 2.454. 
# - Therefore, we can see that loss_test is larger than loss_train, indicating a sign of over-fitting. 

# ## Decision Tree Classification

# **Customer's Eduction**
# - Next, I will use Machine Learning: Decision Tree Classfier in order to use customer's income, generation, and total amount of purchase to predict their education level.

# In[35]:


# Import
from sklearn.tree import DecisionTreeClassifier


# In[36]:


#Normalize the "Education" column
df["Education"]=df["Education"].replace({"Basic":"Undergraduate","2n Cycle":"Undergraduate", "Graduation":"Graduate", "Master":"Postgraduate", "PhD":"Postgraduate"})


# In[37]:


input = ["Income","Total","Generation"]
X =df[input]
y = df["Education"]


# In[38]:


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, random_state=49196138)


# In[47]:


clf = DecisionTreeClassifier(max_leaf_nodes=8)
clf.fit(X_train, y_train) # Fit the classifier to the training data using X for the input features and using "Education" for the target.


# In[40]:


clf.score(X_train, y_train)


# In[41]:


clf.score(X_test, y_test)


# In[42]:


# Illustrate the resulting tree using matplotlib. 
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree


# In[43]:


fig = plt.figure()
_ = plot_tree(clf, 
                   feature_names=clf.feature_names_in_,
                   class_names=clf.classes_,
                   filled=True)


# In[48]:


clf.feature_importances_


# In[50]:


pd.Series(clf.feature_importances_, index=clf.feature_names_in_)


# Feature importance is a score assigned to the features of a Machine Learning model that defines how “important” is a feature to the model's prediction. The feature_importance for Income, Total,and Generation are: 0.462, 0.170, and 0.368. Thus, we can know that "Income" is the most important feature to predict our model's prediction. 

# ## Summary
#     For this project, I first make a graph to show the distribution of the customer's generation. Then I used different regresion, including linear, polynomial, and logitic to show if there's relation between one's income and one's total amount of purchase. And the results showed that there's a position relationship between them. Later, I found out that Income is the the most significant feature to include when I want to predict our model's prediction for education. 

# ## References
# 
# Your code above should include references.  Here is some additional space for references.

# * What is the source of your dataset(s)?

# Customer Personality Analysis: https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis?datasetId=1546318&sortBy=voteCount. 

# * List any other references that you found helpful.

# Reference of code of logic regression: Course Notes from Spring 2022: https://christopherdavisuci.github.io/UCI-Math-10-S22/Week7/Week7-Friday.html .
# Reference of code of K-Nearest Neighbor regression: Course Notes from Winter 2022: 
# https://christopherdavisuci.github.io/UCI-Math-10-W22/Week6/Week6-Wednesday.html . 
# Reference for interactivity altair chart: https://altair-viz.github.io/altair-tutorial/notebooks/06-Selections.html. 
# Reference for Decision Tree Classification and feature_importance: Week 8 Friday lecture: https://deepnote.com/workspace/math-10-f22-9ae9e411-d47b-4572-803e-16ff3e4d5a91/project/Week-8-Friday-12247a05-0b55-4f3c-b8d4-d2ccff50a983/notebook/Week8-Friday-59883ec6c3aa4332a20da4d2653f85e1.
# 

# ## Submission
# 
# Using the Share button at the top right, **enable Comment privileges** for anyone with a link to the project. Then submit that link on Canvas.

# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=e282459e-3f53-4ab2-8b4f-82627aafe86f' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>
