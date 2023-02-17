#!/usr/bin/env python
# coding: utf-8

# # Worksheet 12

# ## Plots of polynomial regression curves
# 
# Write a function `polyreg_plot` which takes four inputs: a pandas DataFrame `df_data` (which can be assumed to have two columns "x" and "y"), an integer `degree` with the default value of `1`, an integer `num_rows` with the default value of `-1`, and an integer `state` with the default value of `0`.  The function should do the following.
# * If `num_rows` is `-1`, the function should set `num_rows` to the number of rows of `df_data`.  (This assignment should go at the top of the body of the function.  Use an if statement.)
# * Select `num_rows` random rows from `df_data` and define `df` to be a copy of these rows.  Use the pandas DataFrame `sample` method with the keyword argument `random_state=state` to select these random rows.  Use the `copy` method to make the copy.
# * For each integer value of `d` from `1` to `degree`, including `degree`, add a new column "xd" to `df` containing the d-th power of the "x" column.  For example, when `d=2`, the column will be named "x2" and will be equal to `df["x"]**2`.  **Comment**.  If you'd rather use `PolynomialFeatures` instead of the "by-hand" method described here, that's great.  I wrote it this way in case we didn't get to `PolynomialFeatures`.
# * Fit a scikit-learn LinearRegression object to this data, using columns "x1" to "xdegree" as the predictor columns and using "y" as the target column.  For example, if `degree=3`, then we would be using the columns "x1", "x2", "x3".  (I believe it's best to make all of the imports outside of the function, so like `from sklearn.linear_model ...` would be written outside of the function, probably in an earlier cell.  The instantiate step, `reg = LinearRegression()` would be written inside the function.)
# * Using NumPy's `arange` function, make a one-dimensional NumPy array containing the values -8, -7.9, -7.8, ..., 7.8, 7.9, and then reshape this array so it has the same length and one column (using `.reshape(???,???)`... you should not need to type any numbers into `reshape` other than $\pm 1$).  Name the resulting "column matrix" as `X1`.
# * Using list comprehension, make a list called `Xlist` containing `X1`, `X1**2`, ..., `X1**degree`.  Concatenate these side-by-side into a NumPy array `X` using `X = np.concatenate(Xlist, axis=???)`.  **Comment**.  Again, you can replace this step with `PolynomialFeatures` if you want.
# * Convert `X` into a pandas DataFrame `dfX` with column names "x1" to "xdegree" as above.  This DataFrame should have 160 rows and degree columns.
# * Use the Linear Regression object you fit above (don't fit it again) to make predictions corresponding to the input `dfX`.  Name the resulting array of predictions `ypred`.  This NumPy array `ypred` should have length 160.
# * Using Matplotlib as in [last week's videos](https://christopherdavisuci.github.io/UCI-Math-10-F22/Week6/Week6-Videos.html?highlight=matplotlib#plotting-using-matplotlib), make a Matplotlib Axes object containing a scatter-plot of the (x,y) points from `df` (use `ax.scatter`) and containing a traditional plot (a line chart, use `ax.plot`) of points (x1, y1) where x1 runs through the "x1" column of `dfX` and where y1 is the corresponding predicted value from `ypred`.  Set the color of the line plot to red by using `'r'` as the format specification. (Even if you're an experienced Matplotlib user, please check the syntax from the video, because I want us to use the Object Oriented Matplotlib syntax, for example, using `plt.subplots` and `ax.scatter` instead of `plt.scatter`.)
# * The function should, as output, return the Matplotlib Figure object.  (As I have it written, the function both shows the plot and also returns the figure object, so don't worry if the figure shows up twice when you call the function.  I'm not sure how to remove the extra display.)

# ## Submission instructions
# 
# * Before evaluating the function, set a Matplotlib style using `plt.style.use(???)` and one of the choices from `plt.style.available`.  I think it makes the most sense to do this outside of the function body.
# * Run the function with `df_data` as the attached `sim_data.csv` (read it in using `pd.read_csv` as usual), with `num_rows=30`, with `degree=7`, and with `state` as your student id number (if you are in a group, just choose one of your student id numbers.)
# * Save the resulting Matplotlib Figure object as a pickle file named *wkst12-ans.pickle*, as in the [Worksheet 1 instructions](https://christopherdavisuci.github.io/UCI-Math-10-F22/Week1/Worksheet1.html), and upload that pickle file to Canvas.
# * If you want to check that your figure has been saved correctly, create a new notebook (or refresh this one) and run the following code (let me know if you see an easier approach):
# 
# ```
# import matplotlib.pyplot as plt
# import pickle
# 
# plt.show()
# 
# with open("wkst12-ans.pickle", "rb") as f:
#     fig = pickle.load(f)
# 
# fig
# ```

# ## Important for Midterm 2
# 
# Not to be turned in but important for the next midterm.  How do the predicted plots (the red curves) change as we use higher degrees?  How do they change as we use smaller or larger subsets of the data (i.e., as we use fewer rows or more rows)?

# ## Reminder
# 
# Every group member needs to submit this on Canvas (even if you all submit the same file).

# ## Submission
# 
# Submit the pickle file on Canvas, as described in the instructions above.
