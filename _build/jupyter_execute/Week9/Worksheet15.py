#!/usr/bin/env python
# coding: utf-8

# # Worksheet 15
# 
# Authors (3 maximum; use your full names): **BLANK**

# In this worksheet, we will use a random forest classifier to classify handwritten digits.
# 
# When we used a single unconstrained decision tree in the [Week 8 videos](https://christopherdavisuci.github.io/UCI-Math-10-F22/Week8/Week8-Videos.html#using-a-decision-tree-with-mnist), we got a test accuracy of 88%.  That is already a good performance, but we will be able to do better using a random forest.

# ## Question 0 - Setting up the workspace
# 
# Make sure you are working in a (free) Pro Deepnote workspace; see the [Worksheet 0 instructions](https://christopherdavisuci.github.io/UCI-Math-10-F22/Week0/Worksheet0.html#question).  The computations are more memory-intensive than usual in this worksheet.  It should say "Pro" or "Education" in the lower-left corner.
# 
# ![Pro logo](../images/pro.png)

# ## Question 1 - Loading the data
# 
# * Load the handwritten digit dataset by executing the following code.  (**Warning**. This dataset is bigger than all of the other datasets we've used in Math 10. If you run this code multiple times, the system may restart.)
# ```
# from sklearn.datasets import fetch_openml
# 
# mnist = fetch_openml('mnist_784')
# ```

# * How many input columns are there?  How many data points are there?  (You might want to review the [Week 8 videos](https://christopherdavisuci.github.io/UCI-Math-10-F22/Week8/Week8-Videos.html) for reminders about this dataset.)

# * Convert the pandas DataFrame `mnist.data` to a NumPy array by using the `to_numpy` method.  The entries are all positive integers between 0 and 255 (inclusive), so convert that NumPy array to an unsigned 8-bit integer dtype by using the NumPy array method `astype` with the argument `np.uint8` (be sure you import NumPy).  Name the resulting NumPy array `X`.  (Having to make these NumPy conversions is a little strange.  I'll point out below which step requires the conversions.)

# * Convert the pandas Series `mnist.target` to a NumPy array again by using the `to_numpy` method, and again convert the data type to unsigned 8-bit integers.  Name the resulting NumPy array `y`.

# * How many bytes is `X`?  Use the `getsizeof` function from the `sys` module.  Make the number easier to read by using an f-string and including commas: `print(???"The NumPy array X is {???:,} bytes.")`.  This is significantly bigger than our usual datasets (but it still probably does not qualify as "big data").

# * Divide `X` and `y` into a training set and a test set, `X_train, X_test, y_train, y_test`, by using `train_test_split` with a `train_size` of `0.8` and a `random_state` of `0`.

# * Check your answer: `y_test` should be a length 14,000 NumPy array with a `dtype` of `uint8`.  The first three values in `y_test` should be `0`, `4`, `1`.

# ## Question 2 - Displaying a handwritten digit
# 
# * We just saw that `y_test[1]` corresponds to the digit `4`.  Display the corresponding handwritten digit from `X_test` using `ax.imshow` where `ax` is a Matplotlib Axes object.  Again, see the Week 8 videos for a reminder of how to do this.  You will need to use `reshape` because the original handwritten digits are 28-by-28 pixels, whereas they are stored in `X_test` as one-dimensional NumPy arrays of length $28^2 = 784$.

# ## Question 3 - Fitting a random forest classifier
# 
# In the third [Week 8 video](https://youtu.be/TOFFhHO_NNk), we used a decision tree classifier with no restrictions and got a test accuracy of 88% (meaning 88% of handwritten digits were correctly identified).  Your goal in this question is to get a test accuracy of at least 92% using a random forest classifier.
# 
# * Import the `RandomForestClassifier` class from scikit-learn's `ensemble` module.
# * Create an instance of this class and name it `rfc` (for random forest classifier).  Experiment with different values of `n_estimators`, `max_depth` and/or `max_leaf_nodes`.  Also use `random_state` to get reproducible results.  Fit the classifier to the training data.  Try to find values which yield a test score (`rfc.score(X_test, y_test)`) of at least `0.92`.  (**Warning**.  Start with small values and work your way up.  If you start with even medium-sized values, the computer may run out of memory and you will have to restart the notebook.  The `fit` step for my values took about 20 seconds.)
# 

# ## Question 4 - The individual trees in the random forest
# 
# * What type of object is `rfc.estimators_`?
# * What is the length of `rfc.estimators_`?  (This and the following questions will only work if you have fit `rfc` in the previous step.)
# * How does the length of `rfc.estimators_` relate to the parameters you used above?
# * What is the type of the zero-th element in `rfc.estimators_`?
# * Using list comprehension, make a list `score_list` containing `clf.score(X_test, y_test)` for each classifier in `rfc.estimators_`.  (This is the step that would not work if we didn't convert to NumPy arrays above.)
# * What is the maximum value in `score_list`?
# * How does this result relate to the expression, "greater than the sum of its parts", or to the phrase, "the wisdom of crowds"?

# ## Question 5 - A DataFrame containing the results
# 
# We now return back to `rfc`.
# 
# * Make a pandas DataFrame df containing two columns.  The first column should be called "Digit" and should contain the values from `y_test`.  The second column should be called "Pred" and should contain the values predicted by `rfc` for the input `X_test`.  (Reality check: `df` should have 14000 rows and 2 columns.)

# ## Question 6 - Confusion matrix
# 
# * Begin making a confusion matrix for this DataFrame using the following code.  This code consists of two Altair charts, a rectangle chart and a text chart.  For now it will look a little strange.
# ```
# import altair as alt
# alt.data_transformers.enable('default', max_rows=15000)
# 
# c = alt.Chart(df).mark_rect().encode(
#     x="Digit:N",
#     y="Pred:N",
# )
# 
# c_text = alt.Chart(df).mark_text(color="white").encode(
#     x="Digit:N",
#     y="Pred:N",
#     text="Pred"
# )
# 
# (c+c_text).properties(
#     height=400,
#     width=400
# )
# ```

# * Specify that the color on the rectangle chart should correspond to `"count()"`, and use one of the [color schemes](https://vega.github.io/vega/docs/schemes/) that seems appropriate, as in `scale=alt.Scale(scheme="???"))`.  (Don't use a categorical scheme and don't use a cyclical scheme.  Try to find a scheme where the differences among the smaller numbers are visible.)
# * You can also add `reverse=True` inside `alt.Scale` if you want the colors to go in the opposite order.  Feel free to change the text color from white if it makes it easier to see.
# * Change the text on the text chart from `"Pred"` to `"count()"`.

# ## Question 7 - Interpreting the confusion matrix
# 
# Use the above confusion matrix to answer the following questions.
# 
# * What is an example of a (true digit, predicted digit) pair that never occurs in the test data?  (Hint/warning.  Pay attention to the order.  The "Digit" column corresponds to the true digit.)
# * What is the most common mistake made by the classifier when attempting to classify a 9 digit?
# * Does that mistake seem reasonable?  Why or why not?
# * Try evaluating the following code.  Do you see why the pandas Series it displays relates to the confusion matrix?
# ```
# df.loc[df["Pred"] == 9, "Digit"].value_counts(sort=False)
# ```

# ## Question 8 - Feature importances
# 
# In general, random forests are more difficult to interpret than decision trees.  (There is no corresponding diagram, for example.)  But random forests can still be used to identify feature importances.
# 
# * Call the `reshape` method on the `feature_imporances_` attribute of `rfc` so that it becomes a 28-by-28 NumPy array.
# * Visualize the result by using the following, where `???` gets replaced by the 28-by-28 NumPy array.  (Here we are using the standard name of `plt` for the Pyplot module.)
# ```
# fig, ax = plt.subplots()
# ax.imshow(???)
# ```
# * Why do you think the pixels around the perimeter are all the same color?

# ## Question 9 - An incorrect digit
# 
# * Find an example of a digit in the test set that was mis-classified by `rfc`.  (Hint.  Start by using `df["Digit"] != df["Pred"]`.  Then you could for example use Boolean indexing, or you could convert to a NumPy array and then use NumPy's `nonzero` function, which will return a length-1 tuple whose only element is a NumPy array containing the integer locations where the value is `True`.)
# * Display the mis-classified handwritten digit using `imshow`.
# * Display the true value (from `y_test` or from `df["Digit"]`).
# * Display the predicted value (using `rfc.predict` or using `df["Pred"]`).
# * Does the mistake by our random forest classifier seem reasonable?  (Some will, some won't.  Find an example where the mistake seems reasonable.)

# ## Reminder
# 
# Every group member needs to submit this on Canvas (even if you all submit the same link).
# 
# Be sure you've included the (full) names of you and your group members at the top after "Authors".  You might consider restarting this notebook (the Deepnote refresh icon, not your browser refresh icon) and running it again to make sure it's working.
# 

# ## Submission
# 
# Using the Share button at the top right, **enable Comment access level** for anyone with the link. Then submit that link on Canvas.

# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=b5575e63-6a41-4f29-bf1d-88636c6d01ec' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>
