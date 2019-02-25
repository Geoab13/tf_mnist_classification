# surpress tf warnings and other verbose
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

mnist = keras.datasets.mnist
(x_train,y_train), (x_test,y_test) = mnist.load_data()

# 0-1 normalize images by dividing with max value of a pixel, 255
# (Min-max normalization since min is 0 in gray-scale images).
x_train, x_test = x_train / 255.0, x_test / 255.0

# plot the 25 first images of the training set
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i], cmap=plt.cm.binary)
    plt.xlabel(y_train[i])
plt.show()

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28), name='input'),
    keras.layers.Dense(50, activation=tf.nn.relu, name='layer_1'),
    keras.layers.Dense(100, activation=tf.nn.relu, name='layer_2'),
    keras.layers.Dense(50, activation=tf.nn.relu, name='layer_3'),
    keras.layers.Dense(10, activation=tf.nn.softmax, name='layer_output')
])

#TensorBoard logger
logger = keras.callbacks.TensorBoard(
    log_dir='keras_log',
    write_graph=True,
    histogram_freq=5
)

#using adam optimizer with sparse_categorical_crossentropy since input is not
# one hot encoded
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#train model by fitting it to the training data, add logger callback
model.fit(
    x_train,
    y_train,
    epochs=40,
    shuffle=True,
    validation_data=(x_test, y_test),
    callbacks=[logger]
    )

#test model
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)

model.save("simple_classification_nn.h5")
