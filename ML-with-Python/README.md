# Machine Learning with Python

These are some notes for the [Machine Learning with Python course](https://www.coursera.org/learn/machine-learning-with-python ), meant to accompany the ipython notebooks

# Week 1 - What is ML?

Major ML techniques:
- Regression/Estimation: for prediciting continuous values
    - e.g. estimating price of house
- Classification: predicting discrete class label or category for a case
   - e.g. is cell cancerous?
- Clustering: finding the structure of data; summarization; segmentation
   - e.g. segmentation of customers
- Associations: associating frequent co-occuring items/events
   - e.g. grocery items that are usually bought together
- Anomaly Detection: discovering abnormal and unusual cases
    - e.g. credit card fraud detection
- Sequence Mining; predicting next events
    - e.g. clickstream -- where will the user next click on a website?
- Dimension Reduction: reducing the size of data 
    - e.g. PCA
- Recommendation Systems: recommending items
    - e.g. spotify music


ML pipeline:

Note: After data preprocessing, we often need to perform feature selection and feature extraction before splitting train/test data.
![](./imgs/ML_pipeline.png)


- ML algorithhimis benefit from standardization of dataset. e.g. can use sklearn.preprocessing.StandardScalar
- can use `svm.SVC` as a classifier
- `sklearn.metrics` has various metrics for determining performance of ML model on test data. e.g. `confusion_matrix`
- can save sklearn models using `pickle`


- column names are called `attributes`
- columns are called `features`
- rows are called `observations`

## Supervised vs unsupervised learning:

### Supervised
How do we *supervise* a ML model? We 'teach the model' by training model with a labelled data-set. The outputs we expect are known. 

- deals with labelled data
- has more evaluation methods than unsupervised learning
- controlled environment

There are two types of supervised learning techniques:
- Classification:
- Regression:

### Unsupervised

Unsupervised learning: let the model discover information on its own. 
data is *unlabelled*

- deals with unlabelled data
- has fewer models and evaluation methods to ensure model is accurate,  than supervised learning
- less controlled environment, since the machine is creating outcomes for us. 

Commonn Unsupervised learning techniques:
- Dimension Reduction / Feature selection: - reduces redundant features/dimensions to make classification easier. 
- Density Estimation: used to explore data to find some structure within it. 
- Market basket analysis: based on theory that if you buy a certain group of items, you're more likely to buy this other group of items
- Clustering

#### Clustering
What is clustering?
- Clustering is grouping of data points or objects that are somehow similar. 
- Used for
   - discovering structure
   - summarization
   - anomaly detection
- most popular unsupervised technique for grouping data


# Week 2 - Regression

- dependant variable must be continuous

## Types of Regression:
- each type below can be linear or non-linear regression


- Simple Regression: only 1 independant variable is used to estimate the dependant
- Multiple regression: more than 1 independant variable is used. Each independant variable will need to be lienarly related to dependant.


## Applications of regression
- sales forecasting
- price estimation
- employment income

## Regression Algorithims:
- Oridnal Regression
- Poisson regression
- Fast Forest quantile Regression
- Linear, Polynomial, Lasoo, Stepwise, Ridge Regression
- Bayesian Linear Regression
- Neural Network regression
- Decision Forest regression
- Boosted decision tree Regression
- KNN (K-nearest neighbors)
   - Although mainly used for classification, [This](https://towardsdatascience.com/the-basics-knn-for-classification-and-regression-c1e8a6c955) describes how KNN can be used for regression

## Evaluation Metrics
- MAE - Mean Absolute Error
- MSE - Mean Squared Error
- RMSE - Square root of MSE. 
- RAE - Relative absolute error / Residual sum of square
- RSE
- R^2 = 1 - RSE

## MLR

How to find coefficents for MLR?
- OLS; minimizes MSE. Uses lin alg operations. Can take a long time for large dataset (>10k rows)
- Optimization algorithim. e.g. Gradient descent. 


Using multiple independant variables can often give better results than SLR. But adding too many independant variables without any real theoretical justification can result in an overfit model (no longer general enough for unseen data)


Explained variance regression score: `= 1- Var(y-y_hat)/Var(y)`
- higher values are better


## Non-linear Regression

### Polynomial Regression
- polynomial regression model can be transformed into a linear regression model, where the `x` variables, simply map to some other non-linear variable. 
- e.g. `x_2 = x^3 in y = m*x + m_2*x_2 + b`
- therefore, can use LeastSquares

### Other non-linear regression
- For a function to be non-linear, `y_hat` must be non-linear function of parameters `w` (coefficents), not necassarily the features `X` 
    - e.g. `y=log(w_0 + w_1*x + w_2*x^2 +..._)`
    - e.g. `y= w_0 + w_1*w_2^x`
- cannot use OLS to fit regression
- estimation of paramters is not easy. 

- for a a sigmoid function (as an example) we can use `scipy.optimize.curve_fit`, which uses non-linear least squares to fit a function, to the data.

# Week 3 - Classification

Classification:
- supervised learning approach
- categorizing items into a discrete set of categories/classes
- target attribute is a *categorical variable*
- e.g. determining if bank customer will default on loan or not (binary classification)
- e.g. Determining appropriate medication for patient with a similar illness (multi-class classification)

## Classification Algorithims:
- Decision Trees
- Naive Bayes
- Linear Discriminant Analysis
- KNN (K-nearest neighbour)
- Logisitic Regression
- SVM (Support vector machines)
- Neural networks

## K-Nearest Neighbours
We choose a class as a predicition by looking at the `K` nearest neighbors

- method for classifying cases based on their similarity to other cases
- KNN approach is based on the fact that similar cases with same class labels are near each other
- cases that are "near" to each other are said to be `neighbours`
- distance between two cases (e.g. Euclidean distance) is a measure of their similarity.

Algorithim:
1. Pick a value of `K`
2. Calculate the distance of unknown case from all (labelled/known) cases
3. Get the `K`-observations in the training data that are "nearest" to the unknown data point.
4. Predict the class of the unknown data point using the most popular class from the `K`-nearest neighbours

Note: out-of-sample data can't be trusted to be used for prediction of unknown samples

```python
from sklearn.neighbors import KNeighborsClassifier
# sklearn's KNN uses Euclidean distance by default
neigh = KNeighborsClassifier(n_neighbors=3) 
neigh.fit(X_train, y_train)
yhat=neigh.predict(X_test)
```

![Plot showing different values of K and their radius](https://res.cloudinary.com/dyd911kmh/image/upload/f_auto,q_auto:best/v1531424125/KNN_final_a1mrv9.png)

There are two slight concerns here:
- how do we select the right `K`?
- how do we calculate similarity / distance between two data points?


### Calculate similarity / distance between two data points
- can use [Minkowski distance](https://en.wikipedia.org/wiki/Minkowski_distance). Generally, we use the [Euclidean distance](http://mathonline.wikidot.com/the-distance-between-two-vectors) form. 
    - each data point is a _vector_, where components of the vector are the data point features (e.g. age, gender, income etc.)
    - find the distance between each vector.
    - Note: need to normalize all features before calculating Euclidean distance. 


There are other dissimilarity measurements that could be used, but it is highly dependant on data type and domain the classification is done for. 


### Selecting the right `K`
if we choose a:
- really low value of `K` (e.g. `K=1`):
    - tend to capture noise or anomolies in data
    - creates highly complex model, would most likely result in overfitting 
    - not generalized enough to be used for out-of sample cases. 
- really high value of `K` (e.g. `K=20`):
    - model becomes overly generalized


So how do we choose a value of `K` then?
- Try training the model for different values of `K`
- determine accuracy of each model varaint on the validation/test data set. 
- Select `K` that gives highest accuracy. 

![](./imgs/knn_accuracy.png)
## Evaluation Metrics in Classification 

- compare actual labels, with predicted labels

### Jaccard Index

- Jaccard = size of intersection, divided by size of union
- a.k.a `IoU = intersection / union`
- higher is better

```python
from sklearn.metrics import jaccard_score¶
```

e.g.
- `y=[0,0,0,0,0,1,1,1,1,1]`
- `y_hat=[1,1,0,0,0,1,1,1,1,1]`
- `IoU = 8/(10+10-8) = 0.66`

![Graphical representation of IoU](https://images.deepai.org/glossary-terms/jaccard-index-452201.jpg)

### F1 Score

In below definitions, we consider the case of binary classification:

When classifying data, we can tally the comparison of predicted label with actual label into groups: True Positive, True Negative, False Positive, and False Negative.


#### Confusion Matrix
A confusion matrix `C` is a square matrix where `Cij` represents the number of instances known to be in group `i` (true label),and were predicted to be in group `j` (predicted label).
- rows show actual labels
- columns show predicted labels by the model
- diagonal elements are correctly predicted labels, while off-diagonal elements are those that were mislabeled by the classifier.
- Note: above definition could be transposed...

In `sklearn`,

```python
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_true, y_pred)
# confusion is a 2D array.
```
See the [sklearn documentation](https://scikit-learn.org/stable/modules/model_evaluation.html#confusion-matrix) for more


![General form of a confusion matrix](https://miro.medium.com/max/492/1*f5ZeXvhsNFZ4q91M4Lotgg.jpeg)

`sklearn`'s `plot_confusion_matrix` gives a plot similar to:

![Confusion matrix plot by sklearn](https://scikit-learn.org/stable/_images/sphx_glr_plot_confusion_matrix_001.png)

#### Accuracy
- number of correctly classified instances over total number of instances
- fraction of total samples that were correctly classified.
- can be misleading for unbalanced datasets, therefore not necassarily always a good measure

```python
from sklearn.metrics import accuracy_score
```

![Equation for Accuracy](https://latex.codecogs.com/png.latex?%5Ctext%7BAccuracy%7D%3D%5Cfrac%7B%5Ctext%7BTN%7D&plus;%5Ctext%7BTP%7D%7D%7B%5Ctext%7BTN%7D&plus;%5Ctext%7BTP%7D&plus;%5Ctext%7BFN%7D&plus;%5Ctext%7BFP%7D%7D)

#### Precision
- Out of all the predicted positive, what percentage is truly positive?
- should ideally be `1`

```python
from sklearn.metrics import precision_score
```

![Equation for Precision](https://latex.codecogs.com/png.latex?%5Ctext%7BPrecision%7D%3D%5Cfrac%7B%5Ctext%7BTP%7D%7D%7B%5Ctext%7BTP%7D&plus;%5Ctext%7BFP%7D%7D)

#### Recall
- Out of all the total positive, what percentage is correctly predicted as positive?
- should ideally be `1`

```python
from sklearn.metrics import recall_score
```

![Equation for Recall](https://latex.codecogs.com/png.latex?%5Ctext%7BRecall%7D%3D%5Cfrac%7B%5Ctext%7BTP%7D%7D%7B%5Ctext%7BTP%7D&plus;%5Ctext%7BFN%7D%7D)

#### F1 Score
- harmonic average of `Precision` and `Recall`
- higher value == higher accuracy. Should ideally be `1`
- shows model has good precision and recall

```python
from sklearn.metrics import f1_score
```

![](https://latex.codecogs.com/png.latex?%5Ctext%7BF1%20Score%7D%3D%5Cfrac%7B2%7D%7B%5Cfrac%7B1%7D%7B%5Ctext%7BPrecision%7D%7D&plus;%5Cfrac%7B1%7D%7B%5Ctext%7BRecall%7D%7D%7D%3D2*%5Cfrac%7B%5Ctext%7BPrecision%7D*%5Ctext%7BRecall%7D%7D%7B%5Ctext%7BPrecision%7D&plus;%5Ctext%7BRecall%7D%7D)

- there is also a _weighted_ F1 Score, in order to give more importance to `Precision` or `Recall`, depending on the requirements. For example, classifying a sick person as healthy would have a different weight as classifying a healthy person as sick.

To find F1 score of a model, 
- calculate for each class 
- Average accuracy of classifier is the average F1 score of all labels (a.k.a Macro F1)

In multiclass clasification, the confusion matrix would look like this:. The green cells are "True Positives" for each class. Thus, `Precision, Recall, F1` etc. is calculated seperately for each class. The example below shows groups of counts, based on if calculating for the `Apple` class.

![Multi-class confusion matrix](./imgs/confusion_matrix.png)

Can easily calculate precision, recall, and f1 score for all classes in multi-class classifier using 
```python
from sklearn.metrics import classification_report
```

From the [sklearn documentation](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-report), it would output something like

```
              precision    recall  f1-score   support

     class 0       0.67      1.00      0.80         2
     class 1       0.00      0.00      0.00         1
     class 2       1.00      0.50      0.67         2

    accuracy                           0.60         5
   macro avg       0.56      0.50      0.49         5
weighted avg       0.67      0.60      0.59         5
```


#### Links:
- https://medium.com/analytics-vidhya/confusion-matrix-accuracy-precision-recall-f1-score-ade299cf63cd
- https://towardsdatascience.com/performance-metrics-confusion-matrix-precision-recall-and-f1-score-a8fe076a2262
- https://towardsdatascience.com/confusion-matrix-for-your-multi-class-machine-learning-model-ff9aa3bf7826
- https://towardsdatascience.com/multi-class-metrics-made-simple-part-i-precision-and-recall-9250280bddc2


### Log Loss
- a.k.a. `Cross-entropy loss`
- for classifiers where the predicted output is a *probability* of the label between 0 and 1, instead of the label itself. 
- increases as predicted probability divergers from actual label (probability of `1`)
- Good classifier models have smaller values of LogLoss.
- e.g. predicting a probability of 0.13 that the data point belongs to class `x`, when the actual label is 1, would be bad, and result in a high log loss.

Let:
- `M` - number of classes (e.g. 3 for (dog, cat, fish))
- `y` - binary indicator (0 or 1) if class label `c` is the correct classification for observation `o` (Label says 'true' probability=1)
- `p` - predicted probability that observation `o` is of class `c`

To find the Log Loss of a model:

1. Calculate log loss for each row

Log Loss equation to measure how far each predicition is from the actual label:

![Log Loss equation](https://latex.codecogs.com/png.latex?-%7B%28y%5Clog%28p%29%20&plus;%20%281%20-%20y%29%5Clog%281%20-%20p%29%29%7D)

Note: the `log` in log loss is actually `ln`. [Since errors are proportional with `ln` vs `log`](https://datascience.stackexchange.com/questions/57009/why-doesnt-the-binary-classification-log-loss-formula-make-it-explicit-that-nat), the base of the logarithim doesn't matter.

2. Then calculate average log loss across all rows of the test set. 

![Total Log Loss equation](https://latex.codecogs.com/gif.latex?-%5Cfrac%7B1%7D%7BM%7D%5Csum_%7Bc%3D1%7D%5EM%7B%28y%5Clog%28p%29%20&plus;%20%281%20-%20y%29%5Clog%281%20-%20p%29%29%7D)

If `M`>2 (i.e. multiclass classification), we calculate a separate loss for each class label per observation and sum the result.

![Multi class logloss](https://latex.codecogs.com/png.latex?-%5Csum_%7Bc%3D1%7D%5EMy_%7Bo%2Cc%7D%5Clog%28p_%7Bo%2Cc%7D%29)


- https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html#loss-cross-entropy
