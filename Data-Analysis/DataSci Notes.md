# Week 1
Data Analsis/Science helps answer questions by looking at data. 

Ex. What should the price of our used car be?

Data Science Thinking:
- Is there data on prices of other cars and their characterisistcs?
- what features of cars affect prices? e.g. color, horsepower, etc. 

features = attributes = characteristics

target value = label = value we want to predict. (Price in this case)


Python Packages in Data Science:
1. Scientific Computing libraries:
- Pandas offers data structure and tools for effective data manipulation and analysis. It provides fast access to structured data. Uses 'dataframes'
- NumPy library uses arrays for its inputs and outputs. Fast Array processing
- Scipy - functions for math problems e.g Integrals, solve Differential Equations, optimization, 
2. Data Visualization
- Matplotlib, Seaborn (based on MatplotLib)
3. Algorithimic Libraries
- Scikit-learn -- ML, regression, classification. Built on NumPy, scipy, matplotlib
- Statsmodels -- explore data, estimate statistical models, and perform statistical tests. 

Read CSV into pandas:
```python
import pandas as pd
# read the online file by the URL provided above, and assign it to variable "df"
path="https://archive.ics.uci.edu/ml/machine-learning-database/autos/imports-85.data"

df = pd.read_csv(path,header=None)
df.head() # or df.info()
```
- can save dataframe as csv file

When doing basic data exploration, check data types in columns of dataframe using `df.dtypes` to make sure it makes sense/ or if it needs to be changed 

Statisical summary: df.describe()
- can also generate statistics for all columns (including string based columns)

Access Database in Python 

Python DB API: (for relational databases)
- Connection Objects -- for DB connections and managing transactions
- Cursor Objects -- for performing DB queries

# Week 2

