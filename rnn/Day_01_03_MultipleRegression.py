# Day_01_03_MultipleRegression.py
import tensorflow as tf


# 문제
# 공부한 시간과 출석한 일수를 반영해서 코드를 재구성하세요
def multiple_regression_1():
    #       1         1        0
    # hx = w1 * x1 + w2 * x2 + b
    # y = x1 + x2
    x1 = [1, 0, 3, 0, 5]    # 공부한 시간
    x2 = [0, 2, 0, 4, 0]    # 출석한 일수
    y = [1, 2, 3, 4, 5]     # 성적

    w1 = tf.Variable(tf.random_uniform([1]))
    w2 = tf.Variable(tf.random_uniform([1]))
    b = tf.Variable(tf.random_uniform([1]))

    hx = w1 * x1 + w2 * x2 + b
    loss = tf.reduce_mean((hx - y) ** 2)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train)
        print(i, sess.run(loss))
    print('-' * 50)

    sess.close()


# 문제
# 피처를 하나로 합쳐주세요
def multiple_regression_2():
    x = [[1, 0, 3, 0, 5],   # 공부한 시간
         [0, 2, 0, 4, 0]]   # 출석한 일수
    y = [1, 2, 3, 4, 5]     # 성적

    w = tf.Variable(tf.random_uniform([2]))
    b = tf.Variable(tf.random_uniform([1]))

    hx = w[0] * x[0] + w[1] * x[1] + b
    loss = tf.reduce_mean((hx - y) ** 2)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train)
        print(i, sess.run(loss))
    print('-' * 50)

    sess.close()


# 문제
# bias를 없애보세요
def multiple_regression_3():
    # x = [[1, 0, 3, 0, 5],   # 공부한 시간
    #      [0, 2, 0, 4, 0],   # 출석한 일수
    #      [1, 1, 1, 1, 1]]
    # x = [[1, 0, 3, 0, 5],   # 공부한 시간
    #      [1, 1, 1, 1, 1],
    #      [0, 2, 0, 4, 0]]   # 출석한 일수
    x = [[1, 1, 1, 1, 1],
         [1, 0, 3, 0, 5],   # 공부한 시간
         [0, 2, 0, 4, 0]]   # 출석한 일수
    y = [1, 2, 3, 4, 5]     # 성적

    w = tf.Variable(tf.random_uniform([3]))

    # hx = w[0] * x[0] + w[1] * x[1] + b
    # hx = w[0] * x[0] + w[1] * x[1] + w[2]
    hx = w[0] * x[0] + w[1] * x[1] + w[2] * x[2]
    loss = tf.reduce_mean((hx - y) ** 2)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(100):
        sess.run(train)
        print(i, sess.run(loss))
    print('-' * 50)

    print(sess.run(w))
    sess.close()


# 문제
# hx를 행렬곱셈으로 수정하세요 (tf.matmul)
def multiple_regression_4():
    x = [[1., 1., 1., 1., 1.],
         [1., 0., 3., 0., 5.],   # 공부한 시간
         [0., 2., 0., 4., 0.]]   # 출석한 일수
    y = [[1, 2, 3, 4, 5]]        # 성적

    w = tf.Variable(tf.random_uniform([1, 3]))

    # (1, 5) = (1, 3) @ (3, 5)
    hx = tf.matmul(w, x)
    loss = tf.reduce_mean((hx - y) ** 2)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train)
        print(i, sess.run(loss))
    print('-' * 50)

    sess.close()


# 문제
# 행렬곱셈에서 x를 앞에 놓으세요
def multiple_regression_5():
    x = [[1., 1., 0.],
         [1., 0., 2.],
         [1., 3., 0.],
         [1., 0., 4.],
         [1., 5., 0.]]
    y = [[1],
         [2],
         [3],
         [4],
         [5]]

    w = tf.Variable(tf.random_uniform([3, 1]))

    # (5, 1) = (5, 3) @ (3, 1)
    hx = tf.matmul(x, w)
    loss = tf.reduce_mean((hx - y) ** 2)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(10):
        sess.run(train)
        print(i, sess.run(loss))
    print('-' * 50)

    sess.close()


# multiple_regression_1()
# multiple_regression_2()
# multiple_regression_3()
# multiple_regression_4()
multiple_regression_5()

print('\n\n\n\n\n\n\n')