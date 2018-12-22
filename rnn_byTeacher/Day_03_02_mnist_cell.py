# Day_03_02_mnist_cell.py
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


def mnist_basic():
    mnist = input_data.read_data_sets('mnist', one_hot=True)

    print(mnist.train.images.shape)         # (55000, 784)
    print(mnist.train.labels.shape)         # (55000, 10)

    print(mnist.validation.images.shape)    # (5000, 784)
    print(mnist.test.images.shape)          # (10000, 784)

    w = tf.Variable(tf.random_uniform([784, 10]))
    b = tf.Variable(tf.random_uniform([10]))

    ph_x = tf.placeholder(tf.float32)
    ph_y = tf.placeholder(tf.float32)

    # (55000, 10) = (55000, 784) @ (784, 10)
    z = tf.matmul(ph_x, w) + b
    hx = tf.nn.softmax(z)
    loss_i = tf.nn.softmax_cross_entropy_with_logits_v2(logits=z, labels=ph_y)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        xx, yy = mnist.train.next_batch(100)
        sess.run(train, {ph_x: xx, ph_y: yy})

        if i % 100 == 0:
            print(i, sess.run(loss, {ph_x: xx, ph_y: yy}))
    print('-' * 50)

    pred = sess.run(z, {ph_x: mnist.test.images})
    pred_arg = np.argmax(pred, axis=1)
    y_arg = np.argmax(mnist.test.labels, axis=1)

    equals = (pred_arg == y_arg)
    print('acc :', np.mean(equals))
    sess.close()


def mnist_rnn_cell():
    mnist = input_data.read_data_sets('mnist', one_hot=True)

    elem_size = 28
    time_step = 28              # sequence_length
    n_classes = 10
    batch_size = 100            # 128
    hidden_size = 150

    ph_x = tf.placeholder(tf.float32, [None, time_step, elem_size])
    ph_y = tf.placeholder(tf.float32, [None, n_classes])

    cell = tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_size)
    outputs, _states = tf.nn.dynamic_rnn(cell, ph_x, dtype=tf.float32)
    print(outputs.shape)        # (100, 28, 150) = (batch_size, time_step, hidden_size)

    # outputs = tf.transpose(outputs, [1, 0, 2])  # (28, 100, 150)
    # final = outputs[-1]

    final = outputs[:, -1, :]       # (100, 150)
    print(final.shape)

    z = tf.layers.dense(final, n_classes)

    loss_i = tf.nn.softmax_cross_entropy_with_logits_v2(logits=z, labels=ph_y)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.RMSPropOptimizer(0.001)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        xx, yy = mnist.train.next_batch(batch_size)
        xx = xx.reshape(batch_size, time_step, elem_size)   # (100, 28, 28)

        sess.run(train, {ph_x: xx, ph_y: yy})

        if i % 100 == 0:
            print(i, sess.run(loss, {ph_x: xx, ph_y: yy}))
    print('-' * 50)

    pred = sess.run(z, {ph_x: mnist.test.images.reshape(-1, time_step, elem_size)})
    pred_arg = np.argmax(pred, axis=1)
    y_arg = np.argmax(mnist.test.labels, axis=1)

    equals = (pred_arg == y_arg)
    print('acc :', np.mean(equals))
    sess.close()





# mnist_basic()
mnist_rnn_cell()









print('\n\n\n\n\n\n\n')