# Day_01_04_LogisticRegression.py
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def show_sigmoid():
    def sigmoid(z):
        return 1 / (1 + np.e ** -z)

    print(np.e)
    print()

    print(sigmoid(-1))
    print(sigmoid(0))
    print(sigmoid(1))

    for z in np.linspace(-5, 5, 30):
        s = sigmoid(z)
        print(z, s)

        plt.plot(z, s, 'ro')
    plt.show()


def logistic_regression():
    x = [[1., 1., 2.],
         [1., 2., 1.],
         [1., 3., 4.],
         [1., 4., 3.],
         [1., 7., 9.],
         [1., 9., 7.]]
    y = [[0],
         [0],
         [0],
         [0],
         [1],
         [1]]
    y = np.int32(y)

    w = tf.Variable(tf.random_uniform([3, 1]))

    # (6, 1) = (6, 3) @ (3, 1)
    z = tf.matmul(x, w)
    hx = tf.nn.sigmoid(z)
    loss_i = y * -tf.log(hx) + (1 - y) * -tf.log(1 - hx)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(100):
        sess.run(train)
        print(i, sess.run(loss))
    print('-' * 50)

    pred = sess.run(hx)
    print(pred)

    pred = pred.reshape(-1)
    print(pred)

    print(pred > 0.5)
    print(np.int32(pred > 0.5))
    sess.close()


# show_sigmoid()

logistic_regression()




print('\n\n\n\n\n\n\n')