# Kaggle - Computer Vision
These are some notes for the [Kaggle - Computer Vision course](https://www.kaggle.com/learn/computer-vision), as summary notes of the ipython notebooks

See the jupyter notebooks for more info.

This course covers:
- building image classifier with Keras
- building CNN with Keras
- How CNN is used for feature extraction
- transfer learning on CNN
- data augmentation

# The Convolutional Classifier

CNN consists of 2 parts: convolutional base, and a dense head.

Base:
- extract features of image. 
- made up of `Conv2D, ReLU, MaxPool2D`

Head:
- determine class of image
- made up of `Dense` and `Dropout` layers

During training, CNN learns:
- which features to extract from an image (base),
- which class goes with what features (head).

## Transfer Learning
Convnets are rarely trained from scratch. More often, we **reuse the base of a pretrained model**. To the pretrained base we then **attach an untrained head**.
- thus needs relatively little data

Example of transfer learning for binary classification
```python
from tensorflow import keras
from tensorflow.keras import layers

pretrained_base = keras.models.load_model(
    '../input/cv-course-models/cv-course-models/vgg16-pretrained-base',
)
pretrained_base.trainable = False

model = keras.Sequential([
    pretrained_base,
    layers.Flatten(),
    layers.Dense(6, activation='relu'),
    layers.Dense(1, activation='sigmoid'),
])
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['binary_accuracy'],
)
```

# Maximum Pooling
Learn more about feature extraction with maximum pooling.

After Convolution+ReLU, there is a lot of large areas with only 0's (black below).

Max Pooling layer condense the feature map to retain only the useful part - the feature. Max Pooling layers condense the input. i.e intensifies features.

![](https://imgur.com/hK5U2cd.png)

the many 0-valued pixels carry only position information. So, `MaxPool2D` i.e. removes some positional info in the feature map. This lets a CNN with max pooling to be invariant to small differences in positions of the features. e.g. framing, perspective, small translation.

`GlobalAvgPool2D` can be used in head as replacement for some of the `Dense` and `Flatten` layers. Also then reduces number of trainable parameters.

# The Sliding Window
Explore two important parameters: stride and padding.

Convolution and MaxPooling both use a sliding window.

Stride
- Convolutional layers will most often have `strides=(1, 1)` because we want high-quality features to use for classification, convolutional layers . Increasing the stride means that we miss out on potentially valuable information in our summary. 
- Maximum pooling layers will almost always have stride values greater than 1, like `(2, 2)` or `(3, 3)`
- Finally, note that when the value of the strides is the same number in both directions

Padding
- `padding='valid'`
    - the convolution window will stay entirely inside the input.
    - drawback is that the output shrinks (loses pixels), and shrinks more for larger kernels. 
    - limits the number of layers the network can contain,
- `padding='same'`. 
    - pads the input with 0's around its borders, using just enough 0's to make the size of the output the same as the size of the input. 
    - has effect of diluting the influence of pixels at the borders.

Example of `padding="same"`, with `stride=1`

<figure>
<img src="https://i.imgur.com/RvGM2xb.gif" width=400 alt="Illustration of zero (same) padding.">
</figure>

# Receptive field
All of the input pixels a neuron is connected to is that neuron's *receptive field*.
- tells which parts of input image a neuron receives info from

Three `(3, 3)` kernels have `27` parameters, while one `(7, 7)` kernel has `49`, though they both create the same receptive field

Stacking of layers create large receptive fields, without increasing number of parameters too much.

# 1D Convolution (Time series)

CNN can be used on time-series data (1-D) and video (3-D data) as well.

Images are 2D, and so a convolution kernels were 2D arrays. 

Time-series is 1D, so kernel is also 1D! - a list!

Example Kernels for 1D time-series data:
```python
detrend = tf.constant([-1, 1], dtype=tf.float32)
average = tf.constant([0.2, 0.2, 0.2, 0.2, 0.2], dtype=tf.float32)
```

- sliding window has only 1 direction to travel (left to right) instead of 2. 

# Custom Convnets
Design your own convnet.

Example CNN
```python
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([

    # First Convolutional Block
    layers.Conv2D(filters=32, kernel_size=5, activation="relu", padding='same',
                  # give the input dimensions in the first layer
                  # [height, width, color channels(RGB)]
                  input_shape=[128, 128, 3]),
    layers.MaxPool2D(),

    # Second Convolutional Block
    layers.Conv2D(filters=64, kernel_size=3, activation="relu", padding='same'),
    layers.MaxPool2D(),

    # Third Convolutional Block
    layers.Conv2D(filters=128, kernel_size=3, activation="relu", padding='same'),
    layers.MaxPool2D(),

    # Classifier Head
    layers.Flatten(),
    layers.Dense(units=6, activation="relu"),
    layers.Dense(units=1, activation="sigmoid"),
])
model.summary()
```
Since the MaxPool2D layer is reducing the size of the feature maps, we can afford to increase the quantity we create, by doubling number of filters at each block.

# Data Augmentation
Boost performance by creating extra training data.
- add in some fake data that looks reasonably like the real data, and classifier will improve
- use reasonable transformations, and don't let transformations mix up classes.

Examples of augmentation
- rotation, color adjustment, contrast adjustment, warping the image

Data augmentation is usually done **online** - i.e. as the images are being fed into the network for training (in mini-batches)

## Keras data augmentation
- can use `ImageDataGenerator` in the data pipeline, or include in model definition using `preprocessing` layers (supported by GPU)
