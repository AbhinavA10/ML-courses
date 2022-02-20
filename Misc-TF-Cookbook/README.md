# Misc. Notes on Keras/TF and a Cookbook

## Approximate Table of Contents
* [Misc Notes](#misc-notes)
* [Common TF Functions](#common-tf-functions)
* [Keras API:](#keras-api-)
* [Cookbook / Keras Recipes](#cookbook---keras-recipes)
* [TensorBoard](#tensorboard)
* [Keras vocab definitions:](#keras-vocab-definitions-)
* [RNN in Keras](#rnn-in-keras)
* [TF Model optimization](#tf-model-optimization)

## Misc Notes

Keras Examples:
- https://keras.io/examples/vision/image_classification_from_scratch/
- https://keras.io/examples/vision/oxford_pets_image_segmentation/
- [Guide to working with tfrecords](https://towardsdatascience.com/ a-practical-guide-to-tfrecords-584536bc786c)
    - https://www.kaggle.com/ryanholbrook/tfrecords-basics/notebook and [accompaying Youtube video](https://www.youtube.com/watch?v=KgjaC9VeOi8)
- [FashionMnist Classification in Keras](https://www.tensorflow.org/tutorials/keras/classification)
- Tensorflow has built some official models here https://github.com/tensorflow/models/tree/master/official.
- In the `resnet` and `mnist` models, they've used the Keras API
    - https://github.com/tensorflow/models/blob/master/official/vision/image_classification/resnet/resnet_model.py
    - https://github.com/tensorflow/models/blob/master/official/vision/image_classification/mnist_main.py
- [StackOverflow - Meaning of None in Keras](https://stackoverflow.com/questions/47240348/-what-is-the-meaning-of-the-none-in-model-summary-of-keras)
    - The first dimension in a keras model is always the batch size, and usually variable
    - `None` means the dimension is variable. e.g. dimension of `(None,255,255,3)`
    -  when you define `input_shape=(100,200)`, actually you're ignoring the batch size and defining the shape of "each sample". Internally the shape will be `(None, 100, 200)`, allowing a variable batch size, each sample in the batch having the shape `(100,200)`.
    - The batch size is automatically defined in the `fit` and `predict` methods.
    - To predict on a single image, would then do something like [this](https://stackoverflow.com/questions/43017017/keras-model-predict-for-a-single-image) and reshape to have `(1,height,width,#channels)`:
    - , in a 2D convolutional network, where the expected input is `(batchSize, height, width, channels)`, you can have shapes like `(None, None, None, 3)`, allowing variable image sizes. 
- https://datascience.stackexchange.com/questions/51829/-having-trouble-understanding-none-in-the-summary-of-my-keras-model


In `Keras` models, to connect many:one layers, you have to be explicit and use a `Concatenate` layer (or an `add` layer, however you want to combine them).
```python
# This works (if same shape)
x0 = Dense(...)
x1 = Dense(...)
x01 = Concatenate()([x0, x1])
y = Dense(...)(x01)
```

- `tf.multiply` === `*` === elementwise multiplication
- `tf.matmul` === `@` === matrix multiplication
- `tf.print(tf.shape(x))` is the way to inspect tensor shapes during runtime
- Can freely mix in tf code between `keras` layers. (in functional api). Lambda layers are not necessary in TF 2.X

Sessions no longer exist in `TensorFlow 2`.
- https://danijar.com/what-is-a-tensorflow-session/
- https://www.tensorflow.org/guide/effective_tf2
- https://blog.tensorflow.org/2019/02/effective-tensorflow-20-best-practices.html
    - If you use the Keras `.fit()` API, you won’t have to worry about dataset iteration. Otherwise need to use `tf.GradientTape()` etc.
    


A past colleague gave me some insight, which I've paraphrased here:

> Q: Tensorflow fully adopted Keras in Tensorflow 2.0. When creating/writing models in “Tensorflow” does anyone use the ‘Tensorflow’ api directly anymore, or is it expected to instead use the Keras API? Is Tensorflow ever used without Keras?


> A: I'd think of them more as different levels. 
>
> `TensorFlow` is a generic mathematical computation library. Think of it as basically equivalent to `numpy` (although certainly with a neural-network emphasis in terms of which operations they have implemented)
>
>e.g. you could write a physics simulation using tensorflow, you could write some signal processing thing, or you could build neural networks, pretty much whatever you want
>
>Keras is, specifically, an API for building and training deep neural networks. and generally speaking if that's what you want to do, you're better off using Keras because it just lets you do much more stuff much more easily than if you were writing it yourself using lower-level `TensorFlow` primitives
>
>That being said, you'll often still be writing parts of your overall pipeline in tensorflow (e.g. loading data, processing outputs, custom loss functions). You can even mix low-level tensorflow ops into your `Keras` model pretty easily, like the following is perfectly valid, and a reasonable thing to do:
>
>```python
>b = tf.keras.layers.Dense(...)(a)  # keras
>c = b * 2                          # TF
>d = tf.keras.layers.Dense(...)(c)  # keras
>```
>
>So overall the answer is that, if you're doing deep learning, certainly use Keras as your main building block, but you'll commonly end up writing some general `TensorFlow` code as well

An interesting image from blog post about what [TF 2.0 includes](https://medium.com/tensorflow/whats-coming-in-tensorflow-2-0-d3663832e9b8):

![](https://miro.medium.com/max/700/0*fJ5u2WE51Oz44dr_)

- [Example of Neural Net in TF 2.0 using the low-level TF API](https://becominghuman.ai/image-classification-with-tensorflow-2-0-without-keras-e6534adddab2)
- [Another example of TF without Keras](https://github.com/AbhinavA10/MachineLearningProjects/blob/master/TensorflowTutorials/04_MNISTRecognition.py)

## Common TF Functions
- [tf.rank(t)](https://www.tensorflow.org/api_docs/python/tf/rank) rank of a tensor is i.e. the # of dimensions in a tensor. Not like rank of a matrix
    - A tensor with batchsize e.g. (2,3,3) is rank 3.

## Keras API:
- https://www.tensorflow.org/guide/keras/sequential_model
- https://www.tensorflow.org/guide/keras/functional
- https://www.tensorflow.org/guide/keras/train_and_evaluate
- https://www.tensorflow.org/guide/keras/save_and_serialize
- https://keras.io/guides/sequential_model/
- https://keras.io/guides/functional_api/
- https://medium.com/@hanify/sequential-api-vs-functional-api-model-in-keras-266823d7cd5e
    - sequential API uses `model.add(layer...)`
    - functional API uses `x=layer(x_0); x_2  = newLayer(x)`. Useful for skip/residual connections --> Concatenate layer
- https://www.tensorflow.org/tutorials/keras/save_and_load#manually_save_weights
- https://keras.io/examples/keras_recipes/debugging_tips/
- https://towardsdatascience.com/debugging-in-tensorflow-392b193d0b8

## Cookbook / Keras Recipes

Can visualize a model using: https://machinelearningmastery.com/visualize-deep-learning-neural-network-model-keras/
1. `print(model.summary())`
2. Plot Keras Model (requires `graphviz` and `pydot`)
```python
 tf.keras.utils.plot_model(
       model,
       to_file="model.png",
       show_shapes=True,
       show_layer_names=True,
)
```

## TensorBoard
- https://keras.io/api/callbacks/tensorboard/ 
- https://www.tensorflow.org/tensorboard/get_started 
- Public Upload for Tensorboard graphs https://tensorboard.dev/#get-started 

## Keras vocab definitions:
- https://keras.io/api/models/model_training_apis/#fit-method 
- https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit 
- `batch_size: int`. Number of samples per gradient update.
- `epochs: int`. Number of epochs to train the model. An epoch is an iteration over the entire training data provided. 
- `steps_per_epoch: int`. Total number of steps (batches of samples) before declaring one epoch finished and starting the next epoch. 

(Note: validation loss of validation data is not affected by regularization layers like noise and dropout.) 

- “batch” is synonymous to a “minibatch”. "Batch size" usually just means mini batch size. Minibatch size is technically the correct term but batch size has been so consistently misused to mean minibatch size that hardly anyone uses the original meaning of batch size (i.e. the size of the whole dataset)


## RNN in Keras
- https://machinelearningmastery.com/rnn-unrolling/ 
- https://www.tensorflow.org/api_docs/python/tf/keras/layers/RNN 
  - input_shape is `[batch_size, timesteps, num_features]`
  - output_shape is  `[batch_size, state_size]` if `return_states`, or `[batch_size, timesteps, output_size]` if `return_sequences`, or `[batch_size, output_size]`
  - `timesteps` is the length of the sequence. 
  - https://stackoverflow.com/questions/47268608/confusion-about-keras-rnn-input-shape-requirement
- `SimpleRNNCell` processes one step within the whole time sequence input, whereas `tf.keras.layer.SimpleRNN` processes the whole sequence. 
- https://keras.io/guides/working_with_rnns/ or https://www.tensorflow.org/guide/keras/rnn 
  - RNN is used for working with time series data, or sequence data
  - Basically, and RNN layer uses a for loop to iterate over the timesteps of a sequence while maintaining an internal state that encodes information about the timesteps it has seen so far
  - For ease of use, use built-in `keras.layers.RNN`, `LSTM`, or `GRU`
  - Output shape is `(batch_size, units)` or if `return_squences` then `(batch_size, timesteps, units)`. `Units` is the `units` argument passed into layer constructor
- https://stackoverflow.com/questions/58748732/dropout-layer-before-or-after-lstm-what-is-the-difference 
  - Using Dropout layer for RNN / LSTM, and how to specify whether to drop timesteps or have consistent dropout mask across timesteps etc. 
  - https://stackoverflow.com/questions/50720670/using-dropout-with-keras-and-lstm-gru-cell 
```python
# Use noise_shape to set dropout mask to be the same for all timesteps
x = tf.keras.layers.Dropout(
  dropout_val, 
  noise_shape=(batch_size, 1, num_features)
  )(x)
```
- To stack LSTM or RNNs in Keras, we need `return_sequences=True` for **all** RNN/LSTM layers except the last one.
  - https://stackoverflow.com/questions/40331510/how-to-stack-multiple-lstm-in-keras 
  - https://machinelearningmastery.com/return-sequences-and-return-states-for-lstms-in-keras/ 
  - https://machinelearningmastery.com/stacked-long-short-term-memory-networks/ 
  - This makes output tensor of earlier RNNs have 3 dimensions tensors, as required for input of downstream RNNs
- In general, `return_sequences = True` is used for either stacking RNNs, or for getting outputs at each timestep. Getting output at each timestep is useful for streaming. Audio classification of a 1 second audio clip doesn’t really need it, but something like predicting things in a continues stream of data would need the output at each timestep
- `StackedRNNCell` makes it more efficient than having separate RNNCells that all output all their timesteps and thus hold their output for all timesteps in memory.

Other notes:
- [https://stats.stackexchange.com/questions/274478/understanding-input-shape-parameter-in-lstm-with-keras](https://stats.stackexchange.com/questions/274478/understanding-input-shape-parameter-in-lstm-with-keras)

- [https://stackoverflow.com/questions/47268608/confusion-about-keras-rnn-input-shape-requirement](https://stackoverflow.com/questions/47268608/confusion-about-keras-rnn-input-shape-requirement)

inputs (x) is shape : # NUM different inputs/sequences, of TIMESTEPS timesteps, and INPUT_D dimensions/features in each input

input shape to model will be input_shape=(TIMESTEPS,INPUT_D)

initial_states of an RNN is usually set to 0.

[https://stackoverflow.com/questions/63044445/setting-the-initial-state-of-an-rnn-represented-as-a-keras-sequential-model](https://stackoverflow.com/questions/63044445/setting-the-initial-state-of-an-rnn-represented-as-a-keras-sequential-model)

[https://towardsdatascience.com/illustrated-guide-to-recurrent-neural-networks-79e5eb8049c9](https://towardsdatascience.com/illustrated-guide-to-recurrent-neural-networks-79e5eb8049c9) 

I saw some code that used the syntax x[[0]] on a 3D numpy array x,

- Why does the output of x[[0]] act like np.expand_dims(x[0], axis=0)?
- What is this syntax called?
- How is a list ([0]) being used as an index of x? i.e. y=[0], print(x[y])

Example snippet:

```python
import numpy as np
rng = np.random.RandomState(3)
x = rng.uniform(-0.5, 0.5, size=(10, 1, 2))
print("x[[0]]", x[[0]], x[[0]].shape)
expanded = np.expand_dims(x[0], axis=0)
print("np.expand_dims(x[0], axis=0)", expanded, expanded.shape)
```
has output
```
x[[0]] [[[0.0507979  0.20814782]]] (1, 1, 2)
np.expand_dims(x[0], axis=0) [[[0.0507979  0.20814782]]] (1, 1, 2)
```

Answer:
you can always use lists as indices to numpy arrays, e.g. x[[0, 3, 5]] will return the 0th, 3rd, and 5th elements from x. so x[[0]] is just a one-element version of that and so if x has shape (n, m), then x[[0, 3, 5]] would have shape (3, m). so by the same reason x[[0]] has shape (1, m). it's called "advanced indexing"

## TF Model optimization
- https://www.tensorflow.org/model_optimization 
  - Collection of TF tools for optimizing ML models for deployment
    - Post-training quantization 
    - Training-time tools
      - Weight Clustering
      - QAT
      - Weight Pruning

- [Inside TensorFlow: TF Model Optimization Toolkit (Quantization and Pruning)](https://www.youtube.com/watch?v=4iq-d2AmfRU) 
- https://www.tensorflow.org/lite/performance/post_training_quantization
