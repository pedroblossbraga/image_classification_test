# Noventiq assingment test Computer Vision

Task: Make a project to classify the presence of St. George in the image.
There are two files with a list of pictures in the folder: with St. Georgies and without.

# Methodology

At first, a simple binary/dichotomous neural network classifier was used, sequentially allocating convolutional layers and max pooling layers. The first layer is a rescaling layer, transforming the interval of images from [0,255] to [0,1], and the last layer has a relu activation function.

Generally, 10 epochs were used, but in a production-model trained in a cloud environment, more epochs could be used.

Since initially the model presented overfitting, data augmentation and dropout regularization were applied. The resulting model had a smooth evolution of training accuracy, and an increase in validation accuracy, whereas the first model had a abrupt increase in training accuracy and a decay in validation accuracy.

In order to use the tf.keras.utils functions to sample data, the images resulting from URLs were stored in a folder for each class.

# References

1. https://www.tensorflow.org/tutorials/images/classification
2. https://medium.com/edureka/tensorflow-image-classification-19b63b7bfd95