# Day_01_05_SoftmaxRegression.py
import numpy as np
import tensorflow as tf


def softmax(z):
    exp = np.exp(z)
    print(exp)

    return exp / np.sum(exp)


def softmax_regression_1():
    x = [[1., 1., 2.],      # C
         [1., 2., 1.],
         [1., 3., 4.],      # B
         [1., 4., 3.],
         [1., 7., 9.],      # A
         [1., 9., 7.]]
    # A B C
    y = [[0, 0, 1],
         [0, 0, 1],
         [0, 1, 0],
         [0, 1, 0],
         [1, 0, 0],
         [1, 0, 0]]
    y = np.int32(y)

    w = tf.Variable(tf.random_uniform([3, 3]))

    # (6, 3) = (6, 3) @ (3, 3)
    z = tf.matmul(x, w)
    hx = tf.nn.softmax(z)
    loss_i = tf.nn.softmax_cross_entropy_with_logits_v2(logits=z, labels=y)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(100):
        sess.run(train)
        print(i, sess.run(loss))
    print('-' * 50)

    sess.close()


# 문제
# 5시간 공부하고 8번 출석한 학생과
# 3시간 공부하고 9번 출석한 학생의 학점을 알려주세요
def softmax_regression_2():
    x = [[1., 1., 2.],      # C
         [1., 2., 1.],
         [1., 3., 4.],      # B
         [1., 4., 3.],
         [1., 7., 9.],      # A
         [1., 9., 7.]]
    # A B C
    y = [[0, 0, 1],
         [0, 0, 1],
         [0, 1, 0],
         [0, 1, 0],
         [1, 0, 0],
         [1, 0, 0]]

    w = tf.Variable(tf.random_uniform([3, 3]))

    ph_x = tf.placeholder(tf.float32)

    # (6, 3) = (6, 3) @ (3, 3)
    z = tf.matmul(ph_x, w)
    hx = tf.nn.softmax(z)
    loss_i = tf.nn.softmax_cross_entropy_with_logits_v2(logits=z, labels=y)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(100):
        sess.run(train, {ph_x: x})
        print(i, sess.run(loss, {ph_x: x}))
    print('-' * 50)

    # pred = sess.run(hx, {ph_x: x})
    pred = sess.run(z, {ph_x: x})
    print(pred)

    pred_arg = np.argmax(pred, axis=1)
    y_arg = np.argmax(y, axis=1)
    print(pred_arg)
    print(y_arg)

    equals = (pred_arg == y_arg)
    print(equals)

    print('acc :', np.mean(equals))
    print('-' * 50)

    result = sess.run(hx, {ph_x: [[1., 5., 8.],
                                  [1., 3., 9.]]})
    print(result)
    print(np.argmax(result, axis=1))
    sess.close()


# print(softmax([2.0, 1.0, 0.1]))

# softmax_regression_1()
softmax_regression_2()

print('\n\n\n\n\n\n\n')