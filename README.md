# <p align="center"> TensorFlow - Simple Neural Network to Classify MNIST Handwritten Digits. </p>
<p align="center"> George Aboalahad </p>

In this TensorFlow program, I construct a simple neural network to classify handwritten digits of the MNIST database. A 3-layer neural network is designed with 50, 100 and 50 nodes in hidden layer 1, 2, and 3, respectively. Further the hidden layers make use of the ReLU activation function, whereas the output layer uses the softmax activation function, to introduce the notion of conditional problabilities. The input images are flattened so the input layer takes batches of shape (?, 784) and are further Min-Max normalized. In this example, full batch optimization is used, meaning that each pass uses the complete training dataset to update the weights.

A part of the input data along with labels used in this program can be seen in the plot below.

![Data](https://github.com/Geoab13/tf_mnist_classification/blob/master/data.png)

In the image below you can see a tensorboard visualization of the computational graph.

![Computational Graph](https://github.com/Geoab13/tf_mnist_classification/blob/master/mnist_computational_graph.png)

## Training and testing

For training 100 epochs was used in this example. For training and testing accuracy and loss, see graphs in the image below. This can easily be changed by updating the num_epochs variable.

![Accuracy](https://github.com/Geoab13/tf_mnist_classification/blob/master/mnist_acc_loss.png)

## Keras Version

In the file *keras-mnist-classifier.py* a corresponding neural network is implemented with less lines of code using the Keras API to tensorflow. Note that under the hood, these neural networks are still different since Keras builds in other functionality and best practices by default. High validation score can be achieved with less epochs.
