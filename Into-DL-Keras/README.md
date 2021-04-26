# Introduction to Deep Learning & Neural Networks with Keras

These are some notes for the [Introduction to Deep Learning & Neural Networks with Keras](https://www.coursera.org/learn/introduction-to-deep-learning-with-keras), meant to accompany the ipython notebooks

 
# Week 1 - Introduction to Deep Learning


Example Applications of Deep Learning
- Colorizing grayscale pictures
- Speech reenactment - synchronzing lip momvments in a video with an audio clip
- Automatic Handwriting Generation (typed text to human handwriting)
- OCR
- Automatically adding sounds to silent movies
- Object classification and detection in images

## Biological Neural Networks

![Biological Neuron Structure](./imgs/biological_neuron.png)

Dendrites:
- receive electrical impulses which carry information, or data, from sensors or synapses of other adjoining neurons
- then carry the impulses or data to the soma. 

Soma
- contains nucleus of neuron.
- In the nucleus, electrical impulses, or the data, are processed by combining them together, and then they are passed on to the axon.  

Axon
- carries the processed information to the synapse, and the output of this neuron becomes the input to thousands of other neurons. 

Learning in the brain occurs by repeatedly activating certain neural connections over others, and this reinforces those connections. This makes them more likely to produce a desired outcome given a specified input. Once the desired outcome occurs, the neural connections causing that outcome become strengthened.

## Artificial Neural Networks
- structured similarily to biological neurons: many inputs/dendrites, a 'nucleus', and a single output / axon.

Layers of Neural Network:

![Layers of Neural Network](./imgs/layers.png)

## Forward Propagation
- process through which data passes through layers of neurons in a neural network from the input layer to the output layer
- neuron output = (linear wieghted sum of inputs + bias) mapped to non linear space using activiation function

Activation function:
- non linear transformation function (e.g. sigmoid function)
- decide whether a neuron should be activated or not, i.e. whether the info the neuron is recieving is relevant or not.
- neural network without activation function is basically just a linear regression model (`y = x1*w1+x2*w2+b1`)

# Week 2 - Artificial Neural Networks

## Gradient Descent

First we define a cost function

Then the optimized weights in the the cost function (minimizing it) are found through *Gradient Descent*

- iterative optimization algoritihim for finding minimum of a function (cost function `f(x)`)
- take steps proportional to negative of the gradient of `f(x)` at the current point (recall Calc 3 / MTE 203)
- size of step is controlled by `learning rate` parameter (eta) -- larger learning rate = bigger steps
- new weights = current weights - learning rate * gradient of cost function @ current wieghts

![Gradient Descent](https://latex.codecogs.com/gif.latex?%5Ctext%7Bnew%20%7Dw_i%20%3D%20%5Ctext%7Bold%20%7Dw_i%20-%20%5Ceta%20%5Ccdot%20%5Cfrac%7B%5Cpartial%20E%7D%7B%5Cpartial%20w_i%7D%20%5Cmid%20%5Ctext%7B@%20old%20%7Dw_i)


## Backpropagation
- this is how NN optimize and learn their weights and biases, through multiple iterations / epochs
- (training is done in supervised learning setting)

Algorithim:
1. Calculate the error (E) between the predicted values and the ground truth labels
    - error = cost function now
    - error usually is MSE, of all predicated outputs compared to all expected ground truths
2. Propagate this error back into the network and use it to perform gradient descent on the different weights and biases in the network. i.e update each weight and bias using:

![Equation for BackPropagation](https://latex.codecogs.com/png.latex?%5C%5C%20w_i%20%5Crightarrow%20w_i%20-%20%5Ceta%20%5Ccdot%20%5Cfrac%7B%5Cpartial%20E%7D%7B%5Cpartial%20w_i%7D%20%5C%5C%20%5C%5C%20%5C%5C%20b_i%20%5Crightarrow%20b_i%20-%20%5Ceta%20%5Ccdot%20%5Cfrac%7B%5Cpartial%20E%7D%7B%5Cpartial%20b_i%7D)

![Algorithim](./imgs/backpropagation.png)

But error is not explicitly a function of w_i or b_i ==> use Chain Rule

![Chainrule](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20E%7D%7B%5Cpartial%20w_3%7D%20%3D%20%5Cfrac%7B%5Cpartial%20E%7D%7B%5Cpartial%20a_3%7D%20%5Ccdot%20%5Cfrac%7B%5Cpartial%20a_3%7D%7B%5Cpartial%20z_3%7D%20%5Ccdot%20%5Cfrac%7B%5Cpartial%20z_3%7D%7B%5Cpartial%20a_2%7D%20%5Ccdot%20%5Cfrac%7B%5Cpartial%20a_2%7D%7B%5Cpartial%20z_1%7D%20%5Ccdot...%20%5Ccdot%20%5Cfrac%7B%5Cpartial%20z_1%7D%7B%5Cpartial%20w_1%7D)


![Training Algorithim](./imgs/training_algo.png)


## Vanishing Gradient Problem
- shortcoming of sigmoid function when used as activation function in the hidden layers of deep networks.
- Gradient of cost function (error) to weights in the earlier layers are very small, since every layer uses the sigmoid function
- Thus neurons in the earlier layers are the slowest to train/learn.
- So, we don't use sigmoid function as activiation functions, since they are prone to vanishing graident problem.

## Activation Functions

Types of Activation Functions
1. Binary Step Function
2. Linear Function
3. Sigmoid / logistic function
4. Hyperbolic Tangent function (tanh)
5. ReLU (Rectified Linear Unit)
6. Leaky ReLU
7. Softmax Function

Generally, should begin with using the ReLU activation function when building a model, and then switch to other activation functions in case ReLU doesn't yield optimum results.

### Sigmoid function 
- pretty flat beyond +-3 reigon ==> small gradient ==> vanishing gradient problem
- avoided in many applications nowadays
- values range from 0 to 1 ==> sigmoid is not symmetric about the origin.
    - sometimes we want different signs for the output of a neuron.

![sigmoid graph](./imgs/sigmoid.png)


### Hyperbolic tangent function
- scaled version of the sigmoid function
- symmetric over the origin.
- still has vanishing gradient problem in very deep neural networks
- avoided in many applications nowadays

![tnah graph](./imgs/tanh.png)

### ReLU
- most widely used activiation function
- non-linear
- does not activate all the neurons at the same time --- if input < 0, neuron will not be activated ==> network will have sparse connections
- overcomes vanishing gradient problem
- only used in hidden layers

![ReLU graph](./imgs/relu.png)

### Softmax function
- type of sigmoid function
- good for classification problems
- ideally used in the output layer of the classifier, where we're actually trying to get the probabilities to get the class of each input.
    - If a network with 3 neurons in the ouput layer outputs `[z1,z2,z3] = [1.66,0.55,0.98]`, with a softmax activation function, the outputs are converted to `[a1,a2,a3] = [0.51,0.18,0.31]`
    
    
![Softmax](https://latex.codecogs.com/gif.latex?a_i%20%3D%20%5Cfrac%7Be%5E%7Bz_i%7D%7D%7B%5CSigma%5E%7Bm%7D_%7Bk%3D1%7De%5E%7Bz_k%7D%7D)


# Week 3 - Deep Learning Libraries
- TensorFlow (used in production)
- Pytorch (used in academic research)
    - cousin to `Torch`
    - allows optimizing custom expressions
- Keras
    - easiest API to use, and go-to library
    - runs on top of Tensorflow

## Keras
- [Models API](https://keras.io/api/models/#about-keras-models)
- [Activation functions available in Keras](https://keras.io/api/layers/activations/)
- [Optimizers available in Keras](https://keras.io/api/optimizers/) - e.g. Gradient Descent
- [Loss functions / Metrics available in Keras](https://keras.io/api/metrics/)

## Regression Models with Keras

Two types of models in Keras: Sequential, and Model class used with functional API

Our model will consist of a linear stack of layers, so Keras' `Sequential` model type can be used

```python
import keras
from keras.models import Sequential
from keras.layers import Dense

model = Sequential() # Create model
n_cols = train_dataset_inputs.shape[1] # Number of predictor variables  / inputs in the dataset

model.add(Dense(5,activation='relu',input_shape=(n_cols,)) # first hidden layer
# After first layer, no longer need to specify input_shape
model.add(Dense(5,activation='relu')) # second hidden layer
model.add(Dense(1) # output layer

# Adam is more efficent than Gradient Descent, and can be used here instead
# Adam does not need a `learning rate` - saves us having to optimize the learning rate for our model
# We will use MSE as the cost function (MSE is for regression)
model.compile(optimizer = 'adam', loss='mean_squared_error') # Setup model parameters
model.fit(predictors,target) # train

predictions = model.predict(test_data) # make new predictions

```

## Classification Models with Keras

- output layer will have n neurons, where n = # of classes that could be output

Suppose target variable column in dataset was something like `[0,0,1,2,3,4,0]`. For training with Keras, this needs to be converted to one-hot encoding / binary class matrix format using
```python
target = to_categorical(dataset['target']) # convert column of data frame into one hot encoding / binary class matrix format
```
```python
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

model = Sequential() # Create model
n_cols = train_dataset_inputs.shape[1] # Number of predictor variables  / inputs in the dataset
target = to_categorical(dataset['target'])

model.add(Dense(5,activation='relu',input_shape=(n_cols,)) # first hidden layer
model.add(Dense(5,activation='relu')) # second hidden layer
# Use softmax so that all outputs in the output layer sum to 1.
model.add(Dense(4, activation='softmax') # output layer

model.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics=['accuracy'])
# we could also define our own metric, and pass it in above, instead of using a built in metric

model.fit(predictors,target, epochs=10) # train

predictions = model.predict(test_data) # make new predictions
# Outputs would be probabilities of the input belonging in a given class (one hot encoded)

```


Week 4 - Deep Learning Models

    Shallow and Deep Neural Networks
    Convolutional Neural Networks
    Recurrent Neural Networks
    Autoencoders
    CNN with Keras