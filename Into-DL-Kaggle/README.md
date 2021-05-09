# Kaggle - Intro to Deep Learning
These are some notes for the [Kaggle - Intro to Deep Learning course](https://www.kaggle.com/learn/intro-to-deep-learning), as summary notes of the ipython notebooks

See the jupyter notebooks for more info.

Deep learning is an approach to machine learning characterized by deep stacks of computations

The first 5 lessons of the course cover using Keras for building regression models.

# A Single Neuron
Learn about linear units, the building blocks of deep learning.

- linear unit is a single linear neuron.
- there are multiple inputs to the neuron
- neuron's output is linear.

Example of a linear model (1 neuron):
```python
# Create a network with 1 linear unit
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    layers.Dense(units=1, input_shape=[3]) # 3 for a vector of length 3
])
```

Model's weights / biasses are initiliazed to random values.

# Deep Neural Networks
Add hidden layers to your network to uncover complex relationships.

- Dense layer = linear neurons with common set of inputs.
- [List of all Keras' built in layers](https://www.tensorflow.org/api_docs/python/tf/keras/layers)
- A "layer" in Keras is any kind of *data transformation*
- two linear dense layers with nothing in between are no better than a single dense layer by itself
- We use activation functions to be able to learn nonlinearties in data
- an *activation* function is applied to a layer's outputs, before the next layer
    - ReLU most common.
    - [List of all Keras' built in activation functions](https://www.tensorflow.org/api_docs/python/tf/keras/activations)
- for regression, the output layer is a linear unit and has no activation function.

E.g. for regression:

```python
model = keras.Sequential([
    # the hidden ReLU layers
    layers.Dense(units=4, activation='relu', input_shape=[2]),
    layers.Dense(units=3),
    layers.Activation('relu'), # can put activation function seperately like this, if wanted
    # the linear output layer 
    layers.Dense(units=1),
])
```

# Stochastic Gradient Descent
Use Keras and Tensorflow to train your first neural network.

    "loss function": measures how good the network's predictions are, compared to expected output.
    "optimizer": tells the network how to change its weights, to minimize loss

All optimization algorithims used in DL belong to a family of `stochastic gradient descent`. - iterative algorithms that train a network in steps. 

One step of training goes like this:
- Sample some training data and run it through the network to make predictions.
- Measure the loss between the predictions and the true values.
- Finally, adjust the weights in a direction that makes the loss smaller.

Then just do this over and over until the loss is as small as you like (or until it won't decrease any further.)

**minibatch**: (often just "**batch**") each iteration's sample of training data

**epoch**: a complete round of the training data. The number of epochs you train for is how many times the network will see each training example

Everytime SGD sees a new minibatch, it updates the weights. 

**learning rate**: A smaller learning rate means the network needs to see more minibatches before its weights converge to their best values

Learning rate and minibatch size have largest effect on how SGD training proceeds
- smaller batch sizes give noisier weight updates and loss curves. This is because each batch is a small sample of data and smaller samples tend to give noisier estimates.
- Smaller learning rates make the updates smaller and the training takes longer to converge. Large learning rates can speed up training, but don't "settle in" to a minimum as well. When the learning rate is too large, the training can fail completely.

`Adam` optimizer is a great SGD algorithm that has an adapttive learning rate. 

**Stochastic** means "determined by chance." Our training is stochastic because the minibatches are random samples from the dataset.

After defining model,
```python
model.compile(
    optimizer="adam",
    loss="mae",
)
```

Training:
```python
history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=256, # feed 256 samples / data points at a time
    epochs=10, # Do it 10 times, all the way through the dataset
)
# convert the training history to a dataframe
history_df = pd.DataFrame(history.history)
# use Pandas native plot method
history_df['loss'].plot();
# history_df.loc[:, ['loss', 'val_loss']].plot();
```
When the loss curve becomes horizontal like that, it means the model has learned all it can and there would be no reason continue for additional epochs

The exercise notebook is a good example of mixing sklearn and tf code to preprocess the data, then using a DNN. 

# Overfitting and Underfitting
Improve performance with extra capacity or early stopping.

Learning Curve with overfitting
![](https://i.imgur.com/tHiVFnM.png)

Training loss will go down either when the model learns *signal* or when it learns *noise*. But the validation loss will go down only when the model learns *signal* (*noise* learned from training set does not generalize to new data. 

Methods to learn signal while reducing noise:
- capacity, early stopping

**Model Capacity**: size and complexity of patterns able to be learnt. Determined by # of neurons and how they are connected. 
- if model is underfitting ==> increase capacity by making wider or deeper
- wider: more neurons in an existing layer. Good for linear relationships
- deeper: more layers. Good for nonlinear relationships

**Early Stopping**: stop training when validation loss no longer decreases.

Early stopping is added through Keras callback, which runs every epoch

```python
from tensorflow.keras.callbacks import EarlyStopping

# If there hasn't been at least an improvement of 0.001 in the 
# validation loss over the previous 20 epochs, then stop the training 
# and keep the best model we found
early_stopping = EarlyStopping(
    min_delta=0.001, # minimium amount of change to count as an improvement
    patience=20, # how many epochs to wait before stopping
    restore_best_weights=True, # Also restores best weights
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    batch_size=256,
    epochs=500,
    callbacks=[early_stopping], # put your callbacks in a list
)
```

# Dropout and Batch Normalization
Add these special layers to prevent overfitting and stabilize training.

Neural networks tend to perform best when their inputs are on a common scale.

Dropout:
- a special type of layer that does not have neurons
- helps avoid overfitting, by randomly droping out some fraction of layers input units every training step. Makes it harder for network to learn the specific "noise" patterns during training. 

Dropout layer is placed before layer dropout should be applied to --- TODO: verify
```python
keras.Sequential([
    # ...
    layers.Dropout(rate=0.3), # apply 30% dropout to the next layer, or previous layer?
    layers.Dense(16),
    # ...
])
```

Batch Normalization:
- helps correct training that is slow or unstable; helps fix problems that may cause training to get "stuck"
- reduces # of epochs needed to complete training.
- batch normalization layer looks at each batch as it comes in, first normalizing the batch with its own mean and standard deviation, and then also putting the data on a new scale with two trainable rescaling parameters
- can be applied to input, which would act like an sklearn `StandardScaler` type thing

```python
layers.Dense(16),
layers.BatchNormalization(),
layers.Activation('relu'),
# layers.BatchNormalization(), # or can place here
```

It's generally a good idea to put all of your data on a common scale. ;SGD will shift the network weights in proportion to how large an activation the data produces. Features that tend to produce activations of very different sizes can make for unstable training behavior.

# Binary Classification
Apply deep learning to another common task.

- uses `Accuracy`to measure success of classification model
    - `accuracy = number_correct / total`
- uses `Cross-entropy` as the loss function (Accuracy since ratio, is a discontinous function)
    - measures the distance from one probability (distribution) to another. 
- layers use `Sigmoid` activation function, rather than ReLU

 For binary classification:
```python
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['binary_accuracy'],
)
```

# Higgs Boson Notebook
The notebook covers:
- using Kaggle's TPU
- parsing `TFRecords` and building a `tf.data.Dataset` object to be used for training
- example of using Keras' Functional API for model building
- using a **Learning rate schedule** to increase or decrease learning rate as needed. (gradually decreasing the learning rate over the course of training can improve performance)
