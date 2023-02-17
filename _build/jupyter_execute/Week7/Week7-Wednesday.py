#!/usr/bin/env python
# coding: utf-8

# # Week 7 Wednesday

# ## Announcements
# 
# * I'm going to move my Wednesday office hours to my office RH 440J (same time: 1pm).
# * The next videos and video quizzes are posted.  Due Monday of Week 8 (because Friday is a holiday this week).
# * Big topics left in the class: overfitting and decision trees/random forests.
# * Midterm: Tuesday of Week 9.  On Wednesday of Week 9 (day before Thanksgiving) I'll introduce the Course Project.  (No Final Exam in Math 10.)

# ## Including a categorical variable in our linear regression
# 
# Here is some of the code from the last class.  We used one-hot encoding to convert the "origin" column (which contains strings) into three separate numerical columns (containing only 0s and 1s, like a Boolean Series).
# 
# We haven't performed the linear regression yet using these new columns.

# In[2]:


import pandas as pd
import numpy as np

import altair as alt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder


# In[3]:


df = sns.load_dataset("mpg").dropna(axis=0)

cols = ["horsepower", "weight", "model_year", "cylinders"]


# In[4]:


reg = LinearRegression()


# In[5]:


reg.fit(df[cols], df["mpg"])


# In[6]:


pd.Series(reg.coef_, index=reg.feature_names_in_)


# In[7]:


encoder = OneHotEncoder()


# In[8]:


encoder.fit(df[["origin"]])


# In[9]:


new_cols = list(encoder.get_feature_names_out())

new_cols


# In[10]:


df2 = df.copy()


# In[11]:


df2[new_cols] = encoder.transform(df[["origin"]]).toarray()


# In[12]:


df2.sample(5, random_state=12)


# This is where the new (Wednesday) material begins.  Let's start with a reminder of what the one-hot encoding is doing.  There are three total values in the "origin" column.  We definitely can't use this column "as is" for linear regression, because it contains strings.  In theory we could replace the strings with numbers, like 0,1,2, but that would enforce an order on the categories, as well as the relative difference between the categories.  Instead, at the cost of taking up more space, we make three new columns.

# In[13]:


df["origin"]


# This particular NumPy array would be two-thirds zeros (why?).  If there were more values, it would be even more dominated by zeros.  For this reason, to be more memory-efficient, by default, scikit-learn saves the result as a special "sparse" object.

# In[14]:


encoder.fit_transform(df[["origin"]])


# We can convert this to a normal NumPy array by using the `toarray` method.

# In[15]:


encoder.fit_transform(df[["origin"]]).toarray()


# Here is a reminder of the last five values in the "origin" column.

# In[17]:


df["origin"][-5:]


# Here are the corresponding rows of the encoding.  Notice how most of the 1 values correspond in the last column (corresponding to "usa"), and the other 1 value is in the first column (corresponding to "europe".  Be sure you understand how the 5 entries in the "origin" column correspond to this 5x3 NumPy array.

# In[16]:


encoder.fit_transform(df[["origin"]]).toarray()[-5:]


# In[18]:


df.columns


# It wouldn't make sense to use  one-hot encoding on the "name" column, because almost every row has a unique name.

# In[19]:


df.name[-5:]


# In[20]:


len(df["name"].unique())


# Let's finally see how to perform linear regression using this newly encoded "origin" column.  We specify that this linear regression object should not learn the intercept.  (We'll say more about why a little later in this notebook.)

# In[21]:


reg2 = LinearRegression(fit_intercept=False)


# We now fit this object using the 4 old columns and the 3 new columns.

# In[23]:


reg2.fit(df2[cols+new_cols], df2["mpg"])


# Because we set `fit_intercept=False`, the `intercept_` value is `0`.

# In[24]:


reg2.intercept_


# More interesting are the seven coefficients.

# In[25]:


reg2.coef_


# Here they are, grouped with the corresponding column names.  Originally I was going to use `index=cols+new_cols`, but this approach, using `index=reg2.feature_names_in_`, is more robust.

# In[26]:


pd.Series(reg2.coef_, index=reg2.feature_names_in_)


# Finding those coefficients is difficult, but once we have the coefficients, scikit-learn is not doing anything fancy when it makes its predictions.  Let's try to mimic the prediction for a single data point.

# In[27]:


df.loc[153]


# With some rounding, the following is the computation made by scikit-learn.  Look at these numbers and try to understand where they come from.  For example, `105` is the "horsepower" value, and `-0.01` is the corresponding coefficient found by `reg2`.
# 
# Notice also the `-17.59` at the end.  This corresponds to the `origin_usa` value.  This ending will always be the same for every car with "origin" as "usa".  This `-17.59` is not being multiplied by anything (or, if you prefer, it is being multiplied by `1`), so it functions like an intercept, a custom intercept for the "usa" cars.  This is the reason why we set `fit_intercept=False` when we created `reg2`.  Another way to look at it, is if we wanted to add for example `13` as an intercept, we could just add `13` to the "origin_europe" value, to the "origin_japan" value, and to the "origin_usa" value; that would have the exact same effect.

# In[1]:


105*-0.01+3459*-0.0057+0.757*75+0.142*6+-17.59


# Let's try to recover that number using `reg2.predict`.  We can't use the following because it is a pandas Series, and hence is one-dimensional.

# In[30]:


df2.loc[153, cols+new_cols]


# By replacing the integer `153` with the list `[153]`, we get a pandas DataFrame with a single row.

# In[31]:


df2.loc[[153], cols+new_cols]


# Now we can use `reg2.predict`.  The resulting value isn't exactly the same as above (`19.19` instead of `19.27`), but I think this distinction is just due to the rounding I did when typing out the coefficients.

# In[32]:


reg2.predict(df2.loc[[153], cols+new_cols])


# ## Polynomial regression
# 
# In linear regression, we find the linear function that best fits the data ("best" meaning it minimizes the Mean Squared Error, discussed more in this week's videos).
# 
# In polynomial regression, we fix a degree `d`, and then find the polynomial function that best fits the data.  We use the same `LinearRegression` class from scikit-learn, and if we use `d=1`, we will get the same results as linear regression.
# 
# * Using polynomial regression with degree `d=3`, model "mpg" as a degree 3 polynomial of "horsepower".  Plot the corresponding predicted values.

# I had intended to use `PolynomialFeatures` from scikit-learn.preprocessing, but because we were low on time, I used a more basic approach.

# In[33]:


# There is a fancier approach using PolynomialFeatures
df.head(3)


# The main trick is to add columns to the DataFrame corresponding to powers of the "horsepower" column.  Then we can perform linear regression.  Phrased another way, if $x_2 = x^2$, then finding a coefficient of $x_2$ is the same as finding a coefficient of $x^2$.
# 
# With a higher degree, you should use a for loop.  I just typed these out by hand to keep things moving faster.

# In[34]:


df["h2"] = df["horsepower"]**2


# In[35]:


df["h3"] = df["horsepower"]**3


# Notice how we have two new columns on the right.  For example, the "h2" value of `16900` in the top row corresponds to `130**2`.

# In[36]:


df.head(3)


# Let's use Linear Regression with these columns (equivalently, we are using polynomial regression of degree 3 with the "horsepower" column).
# 
# It would be more robust to call these instead `["h1", "h2", "h3"]` and to make the list using list comprehension.  That is what you're asked to do on Worksheet 12.

# In[37]:


poly_cols = ["horsepower", "h2", "h3"]


# In[38]:


reg3 = LinearRegression()


# In[39]:


reg3.fit(df[poly_cols], df["mpg"])


# We add the predicted value to the DataFrame.

# In[40]:


df["poly_pred"] = reg3.predict(df[poly_cols])


# Here we plot the predicted values.  Take a moment to be impressed that linear regression is what produced this curve which is clearly not a straight line.  Something that will be emphasized in Worksheet 12, is that as you use higher degrees for polynomial regression, the curves will eventually start to "overfit" the data.  This notion of overfitting is arguably the most important topic in Machine Learning.

# In[41]:


c = alt.Chart(df).mark_circle().encode(
    x="horsepower",
    y="mpg",
    color="origin"
)

c1 = alt.Chart(df).mark_line(color="black", size=3).encode(
    x="horsepower",
    y="poly_pred"
)

c+c1


# Not related to polynomial regression directly, but I wanted to keep the following "starter" code present in the notebook.  It shows an example of specifying a direct numerical value for `y`.  The following could be considered the best "degree zero" polynomial for this data.

# In[3]:


c = alt.Chart(df).mark_circle().encode(
    x="horsepower",
    y="mpg",
    color="origin"
)

c1 = alt.Chart(df).mark_line(color="black", size=3).encode(
    x="horsepower",
    y=alt.datum(df["mpg"].mean())
)

c+c1


# ## Warning: Don't misuse polynomial regression
# 
# For some reason, unreasonable cubic models often get shared in the media.  The cubic polynomial that "fits best" can be interesting to look at, but don't expect it to provide accurate predictions in the future.  (This is a symptom of *overfitting*, which is a key concept in Machine Learning that we will return to soon.)
# 
# In the following, from May 2020, if we trusted the linear fit, we would expect Covid deaths to grow without bound at a constant rate forever.  If we trusted the cubic fit, we would expect Covid deaths to hit zero (and in fact to become negative) by mid-May.

# ![Cubic fit to Covid data](../images/Cubic2.png)

# The following is another example of a cubic fit.  To my eyes, when I look at this data, I do not see anything resemblance to the shown cubic polynomial.  That cubic polynomial is probably the "best" cubic polynomial for this data, but I do not think it is a reasonable model for this data (meaning I don't think any cubic polynomial would model this data well).

# ![Cubic fit to supreme court confirmation votes](../images/Cubic.png)
