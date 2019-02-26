# surpress tf warnings and other verbose
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
mnist = tf.keras.datasets.mnist
import numpy as np

(x_train,y_train), (x_test,y_test) = mnist.load_data()

# 0-1 normalize images by dividing with max value of pixel, 255self (Min-max normalization since min is 0).
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train.reshape([x_train.shape[0], 784])
x_test = x_test.reshape([x_test.shape[0], 784])

#one_hot encode
y_train = np.squeeze(np.eye(10)[y_train.reshape(-1)])
y_test = np.squeeze(np.eye(10)[y_test.reshape(-1)])


# build a NN with 3 layers

learning_rate = 0.001
training_epochs = 100

num_nodes_input = 784
num_nodes_layer1 = 50
num_nodes_layer2 = 100
num_nodes_layer3 = 50
num_nodes_output = 10 #10 class classification (0 - 9)

X = tf.placeholder(shape=(None,num_nodes_input), dtype=tf.float64, name='X')
Y = tf.placeholder(shape=(None,num_nodes_output), dtype=tf.float64, name='Y')

# Layer 1 and using variable_scope to origanize code for tensorboard
with tf.variable_scope('layer1'):
    W1 = tf.get_variable('W1', shape=[num_nodes_input, num_nodes_layer1],\
    dtype=tf.float64, initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable('b1', shape=[num_nodes_layer1],\
    dtype=tf.float64, initializer=tf.zeros_initializer())
    layer1 = tf.nn.relu(tf.add(tf.matmul(X, W1), b1))

# Layer 2
with tf.variable_scope('layer2'):
    W2 = tf.get_variable('W2', shape=[num_nodes_layer1, num_nodes_layer2],\
    dtype=tf.float64, initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable('b2', shape=[num_nodes_layer2], dtype=tf.float64,\
    initializer=tf.zeros_initializer())
    layer2 = tf.nn.relu(tf.add(tf.matmul(layer1, W2), b2))

# Layer 3
with tf.variable_scope('layer3'):
    W3 = tf.get_variable('W3', shape=[num_nodes_layer2, num_nodes_layer3],\
    dtype=tf.float64, initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.get_variable('b3', shape=[num_nodes_layer3], dtype=tf.float64,\
    initializer=tf.zeros_initializer())
    layer3 = tf.nn.relu(tf.add(tf.matmul(layer2, W3), b3))

# output layer
with tf.variable_scope('layeroutput'):
    Wout = tf.get_variable('Wout', shape=[num_nodes_layer3, num_nodes_output],\
    dtype=tf.float64, initializer=tf.contrib.layers.xavier_initializer())
    bout = tf.get_variable('bout', shape=[num_nodes_output], dtype=tf.float64,\
    initializer=tf.zeros_initializer())
    layerout = tf.nn.softmax(tf.add(tf.matmul(layer3, Wout), bout))

#loss function
with tf.variable_scope('loss_function'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=layerout, labels=Y))

#train component
with tf.variable_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

with tf.variable_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(layerout, 1), tf.argmax(Y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

with tf.variable_scope('summary'):
    tf.summary.scalar('current_loss', loss)
    tf.summary.scalar('current_acc', accuracy)
    summary = tf.summary.merge_all()

#now the computational graph is set for training, meaning we can run it in a session
with tf.Session() as session:

    session.run(tf.global_variables_initializer())

    #generate the log files for tensorboard with filewriters
    training_writer = tf.summary.FileWriter("./log/training", session.graph)
    testing_writer = tf.summary.FileWriter("./log/testing", session.graph)

    #we use complete batch training so one epoch is one pass forth and back through the network
    for epoch in range(100):

        # Feed in the training data and do one step of neural network training
        session.run(optimizer, feed_dict={X: x_train, Y: y_train})

        # Every 5 training steps, log our progres
        if epoch % 5 == 0:
            training_loss, training_summary =\
            session.run([loss, summary], feed_dict={X: x_train, Y: y_train})
            testing_loss, testing_summary =\
            session.run([loss, summary], feed_dict={X: x_test, Y: y_test})

            #write data to our log files
            training_writer.add_summary(training_summary, epoch)
            testing_writer.add_summary(testing_summary, epoch)

            print(epoch, training_loss, testing_loss)

    final_training_loss = session.run(loss, feed_dict={X: x_train, Y: y_train})
    final_testing_loss = session.run(loss, feed_dict={X: x_test, Y: y_test})

    print("Final Training cost: {}".format(final_training_loss))
    print("Final Testing cost: {}".format(final_testing_loss))

    print("Accuracy:", accuracy.eval({X: x_test, Y: y_test}))
