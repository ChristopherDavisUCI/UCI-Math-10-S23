#!/usr/bin/env python
# coding: utf-8

# # Worksheet 2
# 
# This worksheet is due Tuesday of Week 2, before discussion section.  You are encouraged to work in groups of up to 3 total students, but each student should submit their own file.  (It's fine for everyone in the group to upload the same file.)
# 
# **Recommendation**.  Follow the Worksheet 0 instructions to form a free Pro workspace for you and your groupmates, so you can all work on the same file.
# 
# These questions refer to the attached vending machines csv file, vend.csv.

# * What are the names of your group members?  Answer as a list of strings.  (Even if you work as a group of 1, still answer as a length-1 list.  You won't get full points if your answer is a string instead of a list.)  Save this list with the variable name `names`.

# In[ ]:





# * We saw on Tuesday several ways to determine how many transaction dates occurred on Saturday, but we didn't see this using the `find` method.  For what value of `z` does the following code produce the same answer?  `(df["TransDate"].str.find("Saturday") != z).sum()` (Hint.  Break this code up into pieces.)

# In[ ]:





# * Preliminary step.  Define a pandas DataFrame `df_june` to be the sub-DataFrame of the full dataset in which the month for the transaction date is June.  (Hints.  Break this up into steps.  Use the pandas `to_datetime` function and the `dt` accessor attribute to find which transaction dates correspond to June.  Then use Boolean indexing to get the corresponding sub-DataFrame.)

# In[ ]:





# * What machine (not to be confused with location) had the most transactions in June?  Save the answer as a string with the variable name `june_mach`.

# In[ ]:





# * What proportion of June transactions is that machine responsible for? Save the answer as a float with the variable name `june_prop`.

# In[ ]:





# * What proportion of overall transactions is that machine responsible for?  Save the answer as a float with the variable name `over_prop`.

# In[ ]:





# * How many transactions were there for which the product name contains the word "Water"?  Don't assume the product name starts or ends with "Water".  (Hint.  Even though Python strings do not have a `contains` method, if `myseries` is a pandas Series of strings, then `myseries.str` does have a `contains` method.)  Save your answer as an integer with the variable name `water`.

# In[ ]:





# * Sort the dataset according to the "Prcd Date" column using the `sort_values` method.    Give an example of two indices `i`,`j` where `df.loc[i, "Prcd Date"]` comes before `df.loc[j, "Prcd Date"]` in this sorted Series, but where the actual date `df.loc[i, "Prcd Date"]` comes after the date `df.loc[j, "Prcd Date"]`.  (*Do not* use `pd.to_datetime`... the point is that the sorting is not correct when we sort as a Series of strings as opposed to a Series of timestamps.)  Save your answer as a tuple `wrong_sort = (i,j)`, where `i` and `j` are integers.  You're expected to solve this by just browsing the sorted Series.  You can try to find these values more systematically, but that's definitely more difficult, and I don't think it's possible with what we've learned so far in Math 10.

# In[ ]:





# * For how many rows in the dataset is the "TransDate" different from the "Prcd Date"?  (Use the `dt` accessor attribute.)  Save your answer as a variable named `diff_dates`.  (Hint.  If your answer is over 6000, something went wrong and you are probably getting that every date is different.)

# In[ ]:





# * Get a link to share this notebook with comment access like on Worksheet 0.  (Click the Share & publish link at the top right; enable public sharing; change the sharing permissions from View to Comment.)  Save this link as a string with the variable name `link`.

# In[ ]:





# 
# 
# 
# 
# 
# * Create a tuple `(names, z, june_mach, june_prop, over_prop, water, wrong_sort, diff_dates, link)` and save this tuple as a pickle file (see the Worksheet 1 instructions).  Submit that pickle file as your Worksheet 2 submission on Canvas.

# In[ ]:





# <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=a0b93ae2-3de8-44e9-bcc0-61282735e4cc' target="_blank">
# <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
# Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>
