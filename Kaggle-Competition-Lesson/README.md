# Kaggle Competition Lessons

Some lessons / tutorial notebooks from Kaggle Courses that are also submissions to Kaggle Competitions, or more real-world examples

# Higgs Boson Notebook
The notebook covers:
- using Kaggle's TPU through a **distribution strategy**, which is like 8 GPUs at once
- parsing `TFRecords` and building a `tf.data.Dataset` object to be used for training
- example of using Keras' Functional API for model building, to build a _wide and deep_ network
- using a **Learning rate schedule** to increase or decrease learning rate as needed. (gradually decreasing the learning rate over the course of training can improve performance)


# Petals to the Metal Notebook
Use Kaggle's TPUs to make a submission to the Petals to the Metal competition

[Petals to the Metal Competition](https://www.kaggle.com/c/tpu-getting-started/overview)

The notebook covers:
- using Kaggle's TPU through a **distribution strategy**
- parsing `TFRecords` and building a `tf.data.Dataset` object to be used for training
- building an image classifier for type of flower, by transfer learning
- a custom learning rate schedule

Through a distribution strategy, TF will distribute the training among the eight TPU cores by creating eight different replicas of the model, one for each core


# Cassava Leaf Disease Classification Notebook
Use Kaggle's TPUs to make a submission to the Cassava Leaf Disease Classification competition.

[Cassava Leaf Disease Classification Competition](https://www.kaggle.com/c/cassava-leaf-disease-classification)

The notebook covers:
- similar items as the `Petals to the Metal` notebook above, but has better explanation on some aspects
