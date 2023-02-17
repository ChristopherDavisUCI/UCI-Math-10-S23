#!/usr/bin/env python
# coding: utf-8

# # Worksheet 8
# 
# This worksheet is due Tuesday of Week 5, before discussion section.  You are encouraged to work in groups of up to 3 total students, but each student should make their own submission on Canvas.  (It's fine for everyone in the group to have the same upload.)

# ## Creating the DataFrame
# 
# * Import the Math2B_grades_clean.csv file.
# * Choose 30 random rows using the pandas DataFrame `sample` method.  Use the keyword argument `random_state` with your student id number, so that the results are always the same.  (If you are in a group, use the student id number of any one of the group members.)  Name this 30-row DataFrame `df30`. 

# * The following image shows the average score for each assignment.  Why would it be difficult to produce the following chart using Altair and `df30`?  (Hint.  What would you put for the x-axis encoding?)
# 
# ![assignment along x-axis](../images/mean_scores.png)

# To fix that, we will create a new longer DataFrame that contains the same data.  
# 
# * Use the pandas DataFrame method `melt` ([documentation](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.melt.html#pandas.DataFrame.melt)) to create a new *variable* column named "Assignment" containing the assignment names and a new *value* column named "Score" containing the scores.  (Do not put the student ids or the final course grades into these new columns.)  Name the resulting DataFrame `df`.

# * Check your answer.  The new DataFrame `df` should have 270 rows and 4 columns.  The columns should be named "Student_id", "Total", "Assignment", "Score".

# * Check your answer.  If you evaluate the following code, you should see something very similar to the above image (but the particular scores will be different because of the randomness).  (If you want to make the chart look even more similar, you can remove the axis titles by using the keyword argument `axis=alt.Axis(title=None)` for both the x-axis and y-axis specifications.)
# 
# ```
# alt.Chart(df).mark_bar().encode(
#     x="Assignment",
#     y="mean(Score)"
# )
# ```

# ## Creating the base chart

# The Altair chart we make will be based on the following.  (Notice it doesn't have a `mark` defined yet, so it will raise an error if you try to plot it.)
# ```
# base = alt.Chart(df).encode(
#     x="Assignment",
#     y="Score",
#     tooltip=["Student_id", "Assignment", "Score"],
# )
# ```

# * Evaluate `base.mark_circle()` to get a sense for the contents.

# * It would be better if the assignments were in the chronological order, rather than alphabetical order.  The following dictionary says what date each assignment occurred on.
# 
# ```
# assignment_dates = {'Final exam': '6/4/22',
#  'Midterm 1': '4/20/22',
#  'Midterm 2': '5/13/22',
#  'Quiz 1': '4/7/22',
#  'Quiz 2': '4/14/22',
#  'Quiz 3': '4/28/22',
#  'Quiz 4': '5/5/22',
#  'Quiz 5': '5/26/22',
#  'Webwork': '4/8/22'}
#  ```

# * Using that dictionary, make a list `assignment_list = ['Quiz 1', 'Webwork', ...]` which lists these assignments in chronological order.  (Here is one approach: convert this dictionary to a pandas Series, then convert the dates using `pd.to_datetime`, then use `sort_values`.  In general in Math 10, it's never correct to just write out all the entries one at a time.)

# * Update the x-channel in the `base` definition to specify `sort=assignment_list` (you will also need to add `alt.X` to input this keyword argument).  Be sure `assignment_list` is really a list, not something like a pandas Index.
# * If you evaluate `base.mark_circle()`, it should look similar to above, but with the x-axis in chronological order.

# ## Creating an interactive chart from the base chart

# Fill in appropriate values to the following template.
# 
# * `???id` should be Student id that you want to highlight.  (This occurs in two places.  Any of the 30 student ids is fine to choose, but choose one for which you think the scores are "interesting" in some way.)
# * `???param` should be the variable name for the selection object.  (This occurs in three places.)
# * `???size` should be the size you want for the highlighted student.
# * `???encode` should be the [encoding abbreviation](https://altair-viz.github.io/user_guide/encoding.html#encoding-data-types) that makes the most sense for student id numbers.
# * Replace `???facet` so that we see a different chart for each course grade.
# * `???description` should be a brief description of something you find interesting about the student you highlighted.
# * `???rs` should be the random state you used at the beginning (in `sample`).
# 
# ```
# highlight = alt.selection(type='single', on='mouseover',
#                           fields=['Student_id'], nearest=True,
#                           init={'Student_id': ???id})
# 
# lines = base.mark_line().encode(
#     size=alt.condition(???param, alt.value(???size), alt.value(1)),
#     color=alt.condition(???param, "Student_id:???encode", alt.value("lightgray"))
# )
# 
# points = base.mark_circle(opacity=0).add_selection(
#     ???param
# )
# 
# chart = alt.layer(lines, points).facet(
#     ???facet
# ).resolve_scale(
#     color="independent"
# ).properties(
#     title={
#       "text": "Student ???id ???description", 
#       "subtitle": "The random_state we used was ???rs"
#     }
# )
# 
# chart
# ```

# ## Reminder
# 
# Every group member needs to submit this on Canvas (even if you all submit the same file).

# ## Submission
# 
# Save `chart` as a json file named "wkst8.json" using the following code, and upload that json file on Canvas.
# 
# ```
# with open("wkst8.json", "w") as f:
#     f.write(chart.to_json())
# ```

# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=fd04a3a4-b4e8-492f-a69c-0f322c4090de' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>
