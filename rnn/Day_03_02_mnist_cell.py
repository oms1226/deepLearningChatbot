#Day_03_02_mnist_cell.py

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def mnist_basic():
    mnist = input_data.read_data_sets('mnist', one_hot=True)

    print(mnist.train.images.shape) #(55000, 784)
    print(mnist.train.labels.shape) #(55000, 10)

    print(mnist.validation.images.shape) #(5000, 784)
    print(mnist.test.images.shape) #(10000, 784)



    w = tf.Variable(tf.random_uniform([784, 10]))#784 feature 의 갯수, 10은 클래스의 갯수
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
        xx, yy = mnist.train.next_batch(100)#100개만큼 데이터를 가져오고, 다 가져오면 셔플해서 가져온다.
        sess.run(train, {ph_x: xx, ph_y: yy})

        if i % 100 == 0:
            print(i, sess.run(loss, {ph_x: xx, ph_y: yy}))
    print('-' * 50)

    # pred = sess.run(hx, {ph_x: x})
    pred = sess.run(z, {ph_x: mnist.test.images})
    pred_arg = np.argmax(pred, axis=1)
    y_arg = np.argmax(mnist.test.labels, axis=1)

    equals = ()
    print(pred_arg)
    print(y_arg)

    equals = (pred_arg == y_arg)
    print('acc :', np.mean(equals))
    print('-' * 50)

    sess.close()


def mnist_rnn_cell():
    mnist = input_data.read_data_sets('mnist', one_hot=True)

    print(mnist.train.images.shape)  # (55000, 784)
    print(mnist.train.labels.shape)  # (55000, 10)

    print(mnist.validation.images.shape)  # (5000, 784)
    print(mnist.test.images.shape)  # (10000, 784)



    ph_x = tf.placeholder(tf.float32, [None, 784])
    ph_y = tf.placeholder(tf.float32, [None, 10])

    z = tf.layers.dense(ph_x, 10)

    loss_i = tf.nn.softmax_cross_entropy_with_logits_v2(logits=z, labels=ph_y)
    loss = tf.reduce_mean(loss_i)

    #optimizer = tf.train.GradientDescentOptimizer(0.1)
    optimizer = tf.train.RMSPropOptimizer(0.001)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        xx, yy = mnist.train.next_batch(100)  # 100개만큼 데이터를 가져오고, 다 가져오면 셔플해서 가져온다.
        sess.run(train, {ph_x: xx, ph_y: yy})

        if i % 100 == 0:
            print(i, sess.run(loss, {ph_x: xx, ph_y: yy}))
    print('-' * 50)

    # pred = sess.run(hx, {ph_x: x})
    pred = sess.run(z, {ph_x: mnist.test.images})
    pred_arg = np.argmax(pred, axis=1)
    y_arg = np.argmax(mnist.test.labels, axis=1)

    equals = ()
    print(pred_arg)
    print(y_arg)

    equals = (pred_arg == y_arg)
    print('acc :', np.mean(equals))
    print('-' * 50)

    sess.close()


# mnist_basic()
mnist_rnn_cell()

print('\n\n\n\n\n')