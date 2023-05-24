# Possible extra topics

One of the rubric items for the course project is to include something "extra" that wasn't covered in Math 10.  Here are a few possibilities.  It's even better if you find your own extra topic; it can be anything in Python that interests you.

More possibilities will be added as I think of them.

## K-Nearest Neighbors

A very nice and understandable Machine Learning model, that I would cover if we had more time, is K-Nearest Neighbors.  There is both a classification version and a regression version.  This topic also provides a good example of the potential for overfitting (when a small number of neighbors is used).  There is some information about this topic in the [course notes from Winter 2022](https://christopherdavisuci.github.io/UCI-Math-10-W22/Week6/Week6-Wednesday.html).

## K-Means Clustering

This is an example of Unsupervised Learning (there are no "correct" labels provided).  I covered this last time I taught Math 10, and you can see the [course notes from Fall 2022](https://christopherdavisuci.github.io/UCI-Math-10-F22/Week5/Week5-Friday.html) (check out the linked Week 5 Friday page as well as the following pages).

## Neural Networks

In both Fall 2021 and Winter 2022, Math 10 included a significant portion on Neural Networks.  These are a fundamental (maybe the most fundamental) area of modern Machine Learning.  If you want to try learning about them, that would be a great extra topic.  In Fall 2021, we used the library TensorFlow (especially using the command `from tensorflow import keras`).  In Winter 2022, we used the library PyTorch.  Overall I think TensorFlow is easier to use (but PyTorch teaches some fundamentals of Object Oriented Programming).

## Principal Component Analysis in scikit-learn

![faces with pca](../images/pca.png)

[Principal component analysis](https://scikit-learn.org/stable/modules/decomposition.html#pca).  Another type of unsupervised learning (along with *clustering*) is *dimensionality reduction*.  Principal Component Analysis (PCA) is a famous example, and it involves advanced linear algebra.  The above image shows a visual example of the result of PCA.

## Choosing parameters

[scikit-learn user guide](https://scikit-learn.org/stable/tutorial/statistical_inference/model_selection.html).  A Machine Learning topic I would like to understand better is how to choose parameters (for example, the number of clusters when doing clustering, or the depth of a decision tree).  That link provides some guidance, but it is a big topic and there are many different approaches.

## pandas styler

![pandas styler](../images/styler.png)

See these examples in the [pandas documentation](https://pandas.pydata.org/pandas-docs/stable/user_guide/style.html#Styler-Functions).  This provides a way to highlight certain cells in a pandas DataFrame, and is good practice using `apply` and `applymap`.

## Kaggle

A general way to get ideas is to browse [Kaggle](https://www.kaggle.com/).  Go to a competition or dataset you find interesting, and then click on the *Code* tab near the top.  You will reach a page like this one about [Fashion-MNIST](https://www.kaggle.com/zalando-research/fashionmnist/code).  Any one of these notebooks is likely to contain many possibilities for extra topics.

## Big Data(sets)

Deepnote does not allow files bigger than 100mb to be uploaded.  Many real-world datasets are bigger than this.  Deepnote does definitely work with larger datasets.  If you end up using a larger dataset, describe how you made it work in Deepnote.  Some general guidelines are listed in the [Deepnote documentation](https://docs.deepnote.com/importing-and-exporing/importing-data-to-deepnote#uploading-files-to-deepnote).

## Different Python libraries

If you want to use a Python library that isn't by default installed in Deepnote, you can install it yourself within Deepnote, using a line of code like the following, which installs the `vega_datasets` library.  Notice the exclamation point at the beginning (which probably won't appear in the documentation you find for the library).
```
!pip install vega_datasets
```

## Other libraries
Here are a few other libraries that you might find interesting.  (Most of these are already installed in Deepnote.)
* [sympy](https://www.sympy.org/en/index.html) for symbolic computation, like what you did in Math 9 using Mathematica.
* [Pillow](https://pillow.readthedocs.io/en/stable/index.html) for image processing.
* [re](https://docs.python.org/3/library/re.html) for advanced string methods using regular expressions.
* [Seaborn](https://seaborn.pydata.org/) and [Plotly](https://plotly.com/python/plotly-express/).  We introduced these plotting libraries briefly together with Altair early in the quarter, and we have used Seaborn frequently for importing datasets.  Their syntax is similar to Altair.
* [ipywidgets](https://ipywidgets.readthedocs.io/en/latest/) provides a way to add interactivity to a Jupyter notebook, but last I checked, not all of it works in Deepnote.

## State-of-the-art Machine Learning
I don't have firsthand experience with these tools, but my understanding is that [XGBoost](https://xgboost.readthedocs.io/en/stable/python/index.html) and [LightGBM](https://lightgbm.readthedocs.io/en/latest/Python-Intro.html) are two of the most successful libraries for modern Machine Learning.  (For example, I think [Jeff Heaton](https://www.youtube.com/channel/UCR1-GEpyOPzT2AO4D_eifdw) mentioned in one of his YouTube videos that these are two of his go-to tools in Kaggle competitions.)  Be sure to make sure they work in Deepnote before investigating too much.

## ChatGPT
I'm not even sure what this would mean, and it wasn't on my radar the last time I taught Math 10 in Fall 22.  One possibility is that you could get help writing your project from ChatGPT, documenting along the way how the process is working.  That would be interesting, just keep it clear what is your work and what is provided by ChatGPT.