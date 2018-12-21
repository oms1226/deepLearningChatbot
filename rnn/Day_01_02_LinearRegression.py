# Day_01_02_LinearRegression.py
import tensorflow as tf


def linear_regression_1():
    # hx = wx + b
    #      1    0
    x = [1, 2, 3]
    y = [1, 2, 3]

    w = tf.Variable(10.)
    b = tf.Variable(10.)

    # hx = tf.add(tf.multiply(w, x), b)
    hx = w * x + b

    # loss = tf.reduce_mean(tf.square(hx - y))
    loss = tf.reduce_mean((hx - y) ** 2)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    train = optimizer.minimize(loss=loss)

    # train = tf.train.GradientDescentOptimizer(0.1).minimize(tf.reduce_mean((tf.add(tf.multiply(w, x), b) - y) ** 2))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    print(w)
    print(sess.run(w))

    for i in range(10):
        sess.run(train)
        # print(i, sess.run(w))
        print(i, sess.run(loss))
    print('-' * 50)

    print(sess.run(hx))
    print(sess.run(w * x + b))
    print(sess.run(w * 5 + b))
    sess.close()


def linear_regression_2():
    x = [1, 2, 3]
    y = [1, 2, 3]

    w = tf.Variable(10.)
    b = tf.Variable(10.)

    ph_x = tf.placeholder(tf.float32)

    hx = w * ph_x + b
    loss = tf.reduce_mean((hx - y) ** 2)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train, feed_dict={ph_x: x})
        print(i, sess.run(loss, {ph_x: x}))
    print('-' * 50)

    # 문제
    # x가 5와 7일 때의 결과를 알려주세요
    print(sess.run(hx, {ph_x: 5}))
    print(sess.run(hx, {ph_x: 7}))
    print(sess.run(hx, {ph_x: x}))
    print(sess.run(hx, {ph_x: [5, 7]}))

    sess.close()


# linear_regression_1()
linear_regression_2()










