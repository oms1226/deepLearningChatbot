# Day_02_03_rnn_3_1.py
import numpy as np
import tensorflow as tf


def show_matmul():
    def tf_matmul(s1, s2):
        m1 = np.prod(s1)
        m2 = np.prod(s2)

        # print(m1, m2)

        aa = np.arange(m1, dtype=np.int32)
        bb = np.arange(m2, dtype=np.int32)

        # print(aa)
        # print(bb)

        a = tf.constant(aa, shape=s1)
        b = tf.constant(bb, shape=s2)

        c = tf.matmul(a, b)
        print('{} @ {} = {}'.format(s1, s2, c.shape))

    tf_matmul([2, 3], [3, 2])
    tf_matmul([3, 2], [2, 4])
    tf_matmul([5, 3, 2], [5, 2, 4])
    # tf_matmul([5, 3, 2], [7, 2, 4])   # error


def show_squeeze():
    x = np.array([[[[[[0], [1], [2]], [[3], [4], [5]]]]]])
    print(x)
    print(x.shape)

    print(x.squeeze(axis=0).shape)
    print()

    # for axis in range(len(x.shape)):
    #     print(axis, np.squeeze(x, axis=axis).shape)

    print(np.squeeze(x))


def show_argmax():
    np.random.seed(12)
    a = np.random.choice(100, 12).reshape(3, 4)
    print(a)
    print()

    print(np.argmax(a))
    print(np.argmax(a, axis=0))     # 수직 (?, 4)
    print(np.argmax(a, axis=1))     # 수평 (3, ?)
    print('-' * 50)

    b = np.random.choice(100, 24).reshape(2, 3, 4)
    print(b)
    print()

    print(b.argmax(0))      # (?, 3, 4)
    print(b.argmax(1))      # (2, ?, 4)
    print(b.argmax(2))      # (2, 3, ?)


def show_onehot():
    a = np.arange(5)
    print(a)

    print(a[0], a[-1])
    print(a[[0, -1]])
    print(a[[0, -1, 2, 3, 0, 0]])   # index array
    print()

    eye = np.eye(5, dtype=np.int32)
    print(eye)
    print()

    print(eye[0])
    print(eye[3])
    print()

    print(eye[[0, 3]])
    print()

    print(eye[[0, 3, 1, 3, 3]])
    print('-' * 50)

    a = np.arange(12).reshape(2, 2, 3)
    print(a)
    print()

    print(a[0])
    print(a[1])
    print()

    print(a[0][0], a[0][1])
    print(a[0, 0], a[0, 1])     # fancy indexing
    print()

    print(a[0][0][0], a[0][1][0])
    print(a[0, 0, 0], a[0, 1, 0])
    print()

    print(a[0, 0], a[0, 1])
    print(a[0, 0, :], a[0, 1, :])
    print(a[0, 0, ::-1], a[0, 1, ::-1])
    print()

    print(a[0, :, 0], a[0, :, 1])


# show_matmul()
# show_squeeze()
# show_argmax()
show_onehot()






print('\n\n\n\n\n\n\n')






