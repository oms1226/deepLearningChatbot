# Day_02_02_rnn_2.py
import numpy as np
import tensorflow as tf

# 문제
# 1번 파일의 코드를 rnn 버전으로 수정하세요
def rnn_2_1():
    vocab = np.array(list('enorst'))

    # tenso
    x = [[0., 0., 0., 0., 0., 1.],
         [1., 0., 0., 0., 0., 0.],
         [0., 1., 0., 0., 0., 0.],
         [0., 0., 0., 0., 1., 0.],
         [0., 0., 1., 0., 0., 0.]]
    # ensor
    y = [[1., 0., 0., 0., 0., 0.],
         [0., 1., 0., 0., 0., 0.],
         [0., 0., 0., 0., 1., 0.],
         [0., 0., 1., 0., 0., 0.],
         [0., 0., 0., 1., 0., 0.]]

    x = np.float32([x])

    hidden_size = 6

    cell = tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_size)
    outputs, _states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
    print(outputs.shape)

    z = outputs[0]

    loss_i = tf.nn.softmax_cross_entropy_with_logits_v2(logits=z, labels=y)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(100):
        sess.run(train)

        pred = sess.run(z)
        pred_arg = np.argmax(pred, axis=1)
        print(i, pred_arg, vocab[pred_arg])

        # print(i, sess.run(loss))
    print('-' * 50)

    print(np.argmax(y, axis=1))
    sess.close()


def rnn_2_2():
    vocab = np.array(list('enorst'))

    # tenso
    x = [[0., 0., 0., 0., 0., 1.],
         [1., 0., 0., 0., 0., 0.],
         [0., 1., 0., 0., 0., 0.],
         [0., 0., 0., 0., 1., 0.],
         [0., 0., 1., 0., 0., 0.]]
    # ensor
    y = [[1., 0., 0., 0., 0., 0.],
         [0., 1., 0., 0., 0., 0.],
         [0., 0., 0., 0., 1., 0.],
         [0., 0., 1., 0., 0., 0.],
         [0., 0., 0., 1., 0., 0.]]

    x = np.float32([x])

    hidden_size = 7

    cell = tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_size)
    outputs, _states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
    print(outputs.shape)    # (1, 5, 2)

    w = tf.Variable(tf.random_uniform([hidden_size, 6]))
    b = tf.Variable(tf.random_uniform([6]))

    # (5, 6) = (5, 2) @ (2, 6)
    z = tf.matmul(outputs[0], w) + b
    # hx = tf.nn.softmax(z)

    loss_i = tf.nn.softmax_cross_entropy_with_logits_v2(logits=z, labels=y)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(100):
        sess.run(train)

        pred = sess.run(z)
        pred_arg = np.argmax(pred, axis=1)
        print(i, pred_arg, vocab[pred_arg])

        # print(i, sess.run(loss))
    print('-' * 50)

    print(np.argmax(y, axis=1))
    sess.close()


def rnn_2_3():
    vocab = np.array(list('enorst'))

    # tenso
    x = [[0., 0., 0., 0., 0., 1.],
         [1., 0., 0., 0., 0., 0.],
         [0., 1., 0., 0., 0., 0.],
         [0., 0., 0., 0., 1., 0.],
         [0., 0., 1., 0., 0., 0.]]
    # ensor
    y = [0, 1, 4, 2, 3]

    x = np.float32([x])

    hidden_size = 7

    cell = tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_size)
    outputs, _states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
    print(outputs.shape)    # (1, 5, 2)

    w = tf.Variable(tf.random_uniform([hidden_size, 6]))
    b = tf.Variable(tf.random_uniform([6]))

    # (5, 6) = (5, 2) @ (2, 6)
    z = tf.matmul(outputs[0], w) + b
    # hx = tf.nn.softmax(z)

    loss_i = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=z, labels=y)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(100):
        sess.run(train)

        pred = sess.run(z)
        pred_arg = np.argmax(pred, axis=1)
        print(i, pred_arg, vocab[pred_arg])

        # print(i, sess.run(loss))
    print('-' * 50)

    print(y)
    sess.close()


# rnn_2_1()
# rnn_2_2()
rnn_2_3()

print('\n\n\n\n\n\n\n')