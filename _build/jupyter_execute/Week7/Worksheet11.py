#!/usr/bin/env python
# coding: utf-8

# # Worksheet 11
# 
# Authors (3 maximum; use your full names): **BLANK**
# 
# This worksheet is due Tuesday of Week 8, before discussion section.  You are encouraged to work in groups of up to 3 total students, but each student should submit their own file.  (It's fine for everyone in the group to upload the same file.)
# 
# **Recommendation**.  Follow the Worksheet 0 instructions to form a free Pro workspace for you and your groupmates, so you can all work on the same file.
# 
# The main part of this week's homework is based on an example from Jake VanderPlas's book, [Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/).

# ## Downloading the data
# 
# Go to [https://data.seattle.gov/](https://data.seattle.gov/) and find the "Fremont Bridge Bicycle Counter" dataset (**not** the one called "Timeline", the plain one).  Download the csv file for that dataset (click the "Export" button at the top right), and upload that csv file to this Deepnote project.  Rename the csv file to "Fremont.csv".  (You can click the three dots to the right of the file name, to reach the option to rename it.  Or just rename it on your computer before you upload the file.)

# ## Linear regression with Seattle bicycle data

# ### Question 1
# 
# * Read in the Fremont.csv file from Part 0, drop the rows which contain missing values, keep only the first two columns, and name the resulting DataFrame `df_pre`.
# * Rename the "Fremont Bridge Total" column to "Bikes", using a command of the form 
# ```
# df_pre.rename({???: ???}, axis=???, inplace=???)
# ```
# * Convert the "Date" column to a `datetime` data type by using `pd.to_datetime`.
# * Using the `dt` accessor and several Boolean Series, define a new pandas DataFrame `df_pre2` which contains only the rows in `df_pre` from the year 2022, from September or earlier, and from the hour 8:00am in the morning.  Use `.copy()` to ensure that `df_pre2` is a new DataFrame.
# * Round the "Date" column to the nearest date (i.e., lose the 8:00am part) by using `dt`, `round` ([documentation](https://pandas.pydata.org/docs/reference/api/pandas.Series.dt.round.html)), and the nearest calendar day [offset](https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases).  (**Updated 11/14**: I had previously said you could use `dt.date` but that fails when we reach the Altair step.  Instead use `dt.round`.)

# ### Question 2
# 
# The weather data in the attached SeattleWeather.csv file was downloaded a few days ago from [this website](https://www.ncdc.noaa.gov/cdo-web/search?datasetid=GHCND).  (You don't need to re-download it; just use the provided csv file in this Deepnote project.)  The "PRCP" column in this csv file indicates the amount of precipitation that fell on that day.
# 
# * Read in the contents of the SeattleWeather.csv file, drop the rows with missing data, and name the result `df_weather`.
# * Rename the "DATE" column to "Date".
# * Convert the "Date" column to a `datetime` data type.
# * Keep only the "Date", "PRCP", and "TMIN" columns in `df_weather`, for example, by using `df_weather = df_weather[???].copy()`.  (The column "PRCP" stands for how much precipitation there was, presumably in inches, and "TMIN" stands for the minimum temperature for that day, in Fahrenheit.)
# * Check your answer: the resulting DataFrame should have 304 rows and 3 columns, and the `dtypes` should be one of `datetime64[ns]` and two of `float64`.

# ### Question 3
# 
# * Use the pandas DataFrame method `merge` ([documentation](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.merge.html)) with `how="inner"` to merge together `df_pre2` and `df_weather` on their "Date" columns.  Name the resulting DataFrame `df`.
# * The resulting DataFrame `df` should have 273 rows and 4 columns.  (**Updated** 11/9: It used to say 5 columns.)

# ### Question 4
# 
# * Add a new column named "DayName" to `df` corresponding to the day of the week (like "Monday") and  a new column called "Month" to `df` that contains the numerical month number (like `11`).  Use the `dt` accessor again.
# * Apply scikit-learn's `OneHotEncoder` class to this "DayName" column.  Name the object `encoder`.
# * Convert the resulting object to a NumPy array using the `toarray` method.  Name the result `arr`.
# * The `encoder.get_feature_names_out()` array will contain strings like `'DayName_Friday'`, `'DayName_Monday'`.  Using list comprehension, make the corresponding list `'Friday'`, `'Monday'`, ...  Make sure your list appears in the same order as `encoder.get_feature_names_out()`.  (One possibility is to use the string method `strip`.  Another possibility is to first `split` the string at `"_"`...)  Call the resulting list `day_list`.
# * Add 7 columns to `df` with the names from `day_list` and put the contents of `arr` into those columns.  (This can be done in a short line of code: `df[???] = ???`.)
# * `df` should now have 13 columns.  (**Updated** 11/9: Used to say 10 columns incorrectly.)

# ### Question 5
# 
# * Instantiate a scikit-learn `LinearRegression` object `reg`.  When you create `reg`, Use the keyword argument `fit_intercept` with an appropriate value to specify that we do not want to find an intercept in this case.  (For this particular data, allowing an intercept value gives no extra flexibility.)
# * Define a list `cols` containing all the numeric columns from `df` except the "Bikes" column.  Use `is_numeric_dtype` from `pandas.api.types` together with list comprehension and `df.columns`.  You can either make a list of all the numeric columns and then remove "Bikes" using the `remove` method, or you can already exclude "Bikes" inside the list comprehension (for example, by using a second condition in the `if` clause).
# * Fit `reg` using `cols` for the input variables, and using "Bikes" for the output variable.
# * Add a "Pred" column to `df` containing the `reg.predict` values.

# ### Question 6
# 
# * Check the values of the fitted coefficients using `pd.Series(???, index=cols)`.
# 
# Answer the following questions in a markdown cell.  (Not as a comment, but as a markdown cell.  You should not need to use the Python comment symbol `#`.  In the course project, you will need to make many of these explanatory markdown cells.)
# 
# * Is the "PRCP" value positive or negative?  Does this make sense?
# * Is the "TMIN" value positive or negative?  Does this make sense?
# * Do the results suggest that people are biking more for exercise or more to get to work/school?  Briefly explain.
# * I expect that if we try the same thing later using also the October, November and December data, that the "Month" coefficient will decrease.  Why do I expect that?

# ### Question 7
# 
# * Make a small pandas DataFrame with only one row, with columns equal to `cols`, and initially filled with all zeros.  Assign it to the variable name `df_test`.  (One approach is to `copy` a one-row sub-DataFrame from `df` with the appropriate columns and then set all values to `0`.  Be sure to use `copy` so that you don't change any values in `df`.  Another approach is to use `pd.DataFrame([???], columns=cols)`, where ??? is a list of zeros of the appropriate length.  Another approach is to use `pd.DataFrame([{???:??? for ??? in cols}])`, where the portion `{???:??? for ??? in cols}` is what's known as a dictionary comprehension; dictionary comprehensions are completely analogous to list comprehensions.)
# * Replace some of the zeros with sample values, making a data point for which our model predicts a negative number of bike riders on December 1st, 2022.  (Don't look up what day of the week that is; use pandas to determine it.)
# * Instead of making a one-row pandas DataFrame, why didn't we just use a pandas Series?  (Hint.  What does scikit-learn expect from the first input to `reg.fit` or from the input to `reg.predict`?)

# ### Question 8
# 
# * Replace the ??? terms so that the following produces the length-7 list `["Monday", "Tuesday", ..., "Sunday"]`.
# ```
# ordered_days = [pd.to_datetime(???"11/???/2022").day_name() for ??? in range(7,???)]
# ```
# 
# * To make an Altair chart `c` containing the actual data in the DataFrame, you can use the following code.
# 
# ```
# sel = alt.selection_single(fields=["DayName"], empty="none")
# 
# base = alt.Chart(df).mark_circle().encode(
#     x="Date",
#     y="Bikes",
#     tooltip=["Bikes", "Date", "DayName", "PRCP", "TMIN"],
#     size=alt.condition(sel, alt.value(60),alt.value(20)),
#     color=alt.Color("DayName", sort=ordered_days)
# ).add_selection(sel)
# 
# text = alt.Chart(df).mark_text(y=20, size=20).encode(
#     text="DayName",
#     opacity=alt.condition(sel, alt.value(1), alt.value(0))
# )
# 
# c = base+text
# ```
# 
# * Define that chart `c` and then display it.
# * Try clicking on one of the points.  What change happens?  (Why are some points selected but not others?)
# * What line of this code guarantees that only the selected day name will be displayed (like "Monday")?
# * Change the base chart so that selected points have an opacity value of `1` and the unselected points have an opacity value of `0.5`.

# ### Question 9
# 
# * Define a second chart `c1` which is a line chart instead of a scatter plot; which uses `color="red"` as a `mark_line()` keyword argument; which again uses "Date" for the x-axis; and which uses the predicted value instead of the actual value for the y-coordinate.
# * Add a `transform_filter` to `c1`, and filter using the selection object created for the `c` chart above.
# * Display a layered chart of `c` and `c1` using `c+c1`.  (If you try to just display `c1` by itself, it won't work, because the selection object was never added to `c1`.)
# * If you try clicking on a point in the chart, you should now see all the points from the same day of the week highlighted, and you should also see a red predicted curve (the "line of best fit") for that day of the week.

# ### Question 10
# 
# * In the predicted curve for Monday, there is an extreme dip near the beginning of March.  What is the cause for this dip?  (Use the tooltip for the corresponding data point.)
# * In the actual data points for Monday, there are some very low values (especially towards the right side of the chart) that aren't matched in the predicted curve.  What do you think is the cause for these low values?  (Hint.  American students might have an unfair advantage on this question.)
# * The red curve represents a linear function, but it certainly doesn't look linear.  Why isn't that a contradiction?
# * Why do the bounds change for the y-axis if you click on a Sunday point?

# ## Reminder
# 
# Every group member needs to submit this on Canvas (even if you all submit the same link).
# 
# Be sure you've included the (full) names of you and your group members at the top after "Authors".  You might consider restarting this notebook (the Deepnote refresh icon, not your browser refresh icon) and running it again to make sure it's working.

# ## Submission
# 
# Using the Share button at the top right, **enable Comment access level** for anyone with the link. Then submit that link on Canvas.
