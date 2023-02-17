#!/usr/bin/env python
# coding: utf-8

# # Worksheet 6
# 
# This worksheet is due Tuesday of Week 4, before discussion section.  You are encouraged to work in groups of up to 3 total students, but each student should make their own submission on Canvas.  (It's fine for everyone in the group to have the same upload.)

# * Import the attached "Math2B_grades_clean.csv" file, and name the DataFrame `df`.
# * Using Boolean indexing, find the sub-DataFrame where the course grade ("Total") is "D" and where the Midterm 1 score is strictly greater than 72.  Name this sub-DataFrame `df_sub`.

# In[ ]:





# * Using Altair, make a scatter plot using the data from `df_sub` for which the x-coordinate is "Midterm 2", the y-coordinate is "Final exam", the color corresponds to the student id number, and the size corresponds to the Quiz 3 score.

# In[ ]:





# * Instead of using the default encoding type for the student id number, specify the encoding type you think makes the most sense for this value.  [Reference](https://altair-viz.github.io/user_guide/encoding.html#encoding-data-types).

# In[ ]:





# * Based on the chart, what are the two student ids of the students who have the lowest scores on Quiz 3?

# In[ ]:





# * Add a tooltip to the chart so that you can tell exactly what the student id and the Quiz 3 score are for each point.  Do your identified students indeed have the lowest Quiz 3 scores?

# In[ ]:





# * Here is a way to find one of those student ids (not both) using pandas.  Can you figure out how the following code works by breaking it up into pieces?  (There might be a question based on this code on the next quiz or on the midterm.) 
# ```
# df_sub.set_index("Student_id")["Quiz 3"].idxmin()
# ```

# In[ ]:





# * Why does the following code give a different answer?
# ```
# df_sub["Quiz 3"].idxmin()
# ```

# In[ ]:





# * What changes if we use `argmin` instead of `idxmin`?  What is the difference between these two methods?

# In[ ]:





# * We were just talking about two students who had the same score on Quiz 3.  Can you identify these same two students from the following Altair chart?  (Notice how we have moved back from `df_sub` to `df`.  Scroll all the way to the right to see the color scale.)
# ```
# alt.Chart(df).mark_circle().encode(
#     x=alt.X("Midterm 1", scale=alt.Scale(zero=False)),
#     y="Midterm 2",
#     color=alt.Color("Quiz 3", scale=alt.Scale(scheme="turbo")),
#     tooltip=["Student_id", "Midterm 1"]
# ).facet("Total").resolve_scale(
#     x='independent'
# )
# ```

# In[ ]:





# * Try to make the above FacetChart using the following template.  (There will be slight differences; for example, the chart titles will be more bold in this version.)
# ```
# chart_list = []
# 
# for grade, df_mini in df.groupby(???):
#     c = alt.Chart(???).mark_circle().encode(
#         x = alt.X("Midterm 1", scale=alt.Scale(zero=False)),
#         y = "Midterm 2",
#         color=alt.Color("Quiz 3", scale=alt.Scale(scheme="turbo")),
#         tooltip=["Student_id", "Midterm 1"]
#     ).properties(
#         title=???
#     )
#     chart_list.append(???)
# 
# alt.hconcat(*???)
# ```

# In[ ]:





# * If `mychart` is an Altair chart object (like the output of `alt.hconcat` in the previous question), you can save the chart as a json file using the following code.   Create this json file for the `HConcatChart` object you just created, and upload that json file to Canvas.
# ```
# with open("hconcat_file.json", "w") as f:
#     f.write(mychart.to_json())
# ```

# In[ ]:





# * If you want to check that your json file contains the correct information, you can try pasting its contents into the [Vega-Lite editor](https://vega.github.io/editor/#/edited) and then clicking Run at the top.  (You should see a warning about the Vega-Lite version, but the chart itself should look correct.)  Or you can create a new notebook, load the json file using
# ```
# with open("hconcat_file.json", "r") as f:
#     s = f.read()
# ```
# and then use `alt.Chart.from_json(???)`.

# In[ ]:





# ## Reminder
# 
# Every group member needs to submit this on Canvas (even if you all submit the same file).
# 
# Be sure you've included the names of you and your group members at the top after "Authors".  You might consider restarting this notebook (the Deepnote refresh icon, not your browser refresh icon) and running it again to make sure it's working.
