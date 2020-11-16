# Data Visualization with Python
These are some notes for the [Data Visualization with Python course](https://www.coursera.org/learn/python-for-data-visualization), meant to accompany the ipython notebooks
This course was pretty basic. The videos didn't cover much more than already covered in past courses. The labs offered more in terms of creating visualizations. 

I also used this course to learn more about visualization techniques on my own. 

https://python-graph-gallery.com/
- This site is a collection of good graphs made with seaborn, matplotlib, etc. with reference code. 

2D Density Plot / Hexbin Plot
- useful to represent the relationship of 2 numerical variables when you have a lot of data points
[![2d density plot](https://python-graph-gallery.com/wp-content/uploads/86_2D_density_plot_explanation.png)](https://python-graph-gallery.com/2d-density-plot/)

Boxplots:
- although a good summary, we cannot see what is the underlying **distribution of dots** in each group, or the **number of observations** for each.
- To solve this, we can add a stripplot as shown [here](https://python-graph-gallery.com/39-hidden-data-under-boxplot/): 

```python
ax = sns.boxplot(x='group', y='value', data=df)
ax = sns.stripplot(x='group', y='value', data=df, color="orange", jitter=0.2, size=2.5)
plt.title("Boxplot with jitter", loc="left")
```

![Boxplot with jitter](https://python-graph-gallery.com/wp-content/uploads/39_Bad_boxplot2.png)

Matplot lib has [various styles](https://matplotlib.org/3.1.1/gallery/style_sheets/style_sheets_reference.html), and plots can be [annotated](https://matplotlib.org/tutorials/text/annotations.html)

## Pandas GroupBy

From lab3: 

The general process of `groupby` involves the following steps:

1.  **Split:** Splitting the data into groups based on some criteria.
2.  **Apply:** Applying a function to each group independently:
       .sum()
       .count()
       .mean() 
       .std() 
       .aggregate()
       .apply()
       .etc..
3.  **Combine:** Combining the results into a data structure.
<img src="https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DV0101EN/labs/Images/Mod3Fig4SplitApplyCombine.png" height=400 align="center">

```python
# group countries by continents and apply sum() function 
df_continents = df_can.groupby('Continent', axis=0).sum()
# note: the output of the groupby method is a `groupby' object. 
# we can not use it further until we apply a function (eg .sum())
df_continents.head()
```

# Week 1 - Introduction to Data Visualization Tools

In a jupyter notebook, we generally do:
```python
import matplotlib.pyplot as plt
%matplotlib inline
plt.plot(...)
plt.show()
plt.title("Title") # doesn't work, needs to be before `show()`
```

We are able to make more editable figures, using the `notebook` backend. This backend checks if a figure has already been made, and if so, simply edits it
```python
# %matplotlib inline
%matplotlib notebook
plt.plot(...)
plt.show()
plt.title("Title") # works
```

Pandas has some built-in matplotib functions:
- `df.plot(kind='line')`
- `df['col_name'].plot(kind='hist')`

# Week 2 - Basic and Specialized Visualization Tools

- can transpose a pandas dataframe using `df.transpose()`

Histogram
- sometimes tick marks may not align with bins. Can use `numpy` to calculate bin marking:
```python
count, bin_edges = np.histogram(df['col_name'])
df['col_name'].plot(kind='hist', xticks = bin_edges)
```

**Subplots** - From Lab 3:

Often times we might want to plot multiple plots within the same figure. For example, we might want to perform a side by side comparison of the box plot with the line plot of China and India's immigration.

To visualize multiple plots together, we can create a **`figure`** (overall canvas) and divide it into **`subplots`**, each containing a plot. With **subplots**, we usually work with the **artist layer** instead of the **scripting layer**. 

Typical syntax is : <br>

```python
    fig = plt.figure() # create figure
    ax = fig.add_subplot(nrows, ncols, plot_number) # create subplots
    
```

Where

-   `nrows` and `ncols` are used to notionally split the figure into (`nrows` * `ncols`) sub-axes,  
-   `plot_number` is used to identify the particular subplot that this function is to create within the notional grid. `plot_number` starts at 1, increments across rows first and has a maximum of `nrows` * `ncols` as shown below.

<img src="https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DV0101EN/labs/Images/Mod3Fig5Subplots_V2.png" width=500 align="center">

# Week 3 - Advanced Visualizations and Geospatial Data

Waffle Charts
- good for showing how much a specific category makes up the entire dataset
![](https://raw.githubusercontent.com/gyli/PyWaffle/aec778fed3827d35ce104e1582dda6a38c5ed59f/examples/readme/title_and_legend.svg)
- can use Function created in `Lab 4`, or this library; [Pywaffle](https://github.com/gyli/PyWaffle)

Word Clouds
- use [Word_cloud](https://github.com/amueller/word_cloud/) library

![Example of wordcloud](https://raw.githubusercontent.com/amueller/word_cloud/master/examples/alice.png)

Folium
- python library to visualize geo-spatial data.
- can superimpose heatmaps, colors, markers onto a map. 

For more, see https://www.kaggle.com/learn/geospatial-analysis or https://www.kaggle.com/alexisbcook/interactive-maps

