# <p align="center"> TensorFlow - Simple Neural Network to Classify MNIST Handwritten digits. </p>

In this TensorFlow program, I construct a simple neural network to classify handwritten digits of the MNIST database. A 3-layer neural network is designed with 50, 100 and 50 nodes in hidden layer 1, 2, and 3, respectively. Further the hidden layers make use of the ReLU activation function, whereas the output layer uses the softmax activation function, to introduce the notion of conditional problabilities. The input images are flattened so the input layer takes batches of shape (?, 784) and are further Min-Max normalized. In this example, full batch optimization is used, meaning that each pass uses the complete training dataset to update the weights.

In the image below you can see a tensorboard visualization of the computational graph.

![alt text](https://raw.githubusercontent.com/username/projectname/branch/path/to/img.png)

## Training and testing

For training 100 epochs was used in this example. For training and testing accuracy and loss, see graphs in the image below. This can easily be changed by updating the num_epochs variable.
