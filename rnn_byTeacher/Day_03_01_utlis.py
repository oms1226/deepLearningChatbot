# Day_03_01_utlis.py
import numpy as np
import tensorflow as tf


def show_lambda():
    def twice(n):
        return n * 2

    def proxy(func, n):
        print(func)
        print(func(n))

    lamb = lambda n: n * 2

    f = twice
    print(f)
    print(twice)

    print(twice(3))
    print(f(3))
    print(lamb(7))
    print((lambda n: n * 2)(7))

    proxy(twice, 3)
    proxy(lambda n: n * 2, 3)


def show_map():
    def square(n):
        return n * n

    inputs = np.array([0, 1, 2, 3, 4, 5])

    # map_op = tf.map_fn(square, inputs)
    map_op = tf.map_fn(lambda n: n * n, inputs)

    with tf.Session() as sess:
        print(sess.run(map_op))


def show_scan():
    inputs = np.array(['1', '2', '3', '4', '5'])

    scan_op = tf.scan(lambda a, n: a + n, inputs)

    with tf.Session() as sess:
        print(sess.run(scan_op))


# show_lambda()
# show_map()
# show_scan()


