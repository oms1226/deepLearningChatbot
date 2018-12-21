# Day_02_03_rnn_3_1.py
import platform
import numpy as np
import tensorflow as tf

def show_matmul():
    def tf_matmul(s1, s2):
        m1 = np.prod(s1)
        m2 = np.prod(s2)

        print (m1, m2)

        aa = np.arange(m1, dtype=np.int32)
        bb = np.arange(m2, dtype=np.int32)

        a = tf.constant(aa, shape=s1)
        b = tf.constant(bb, shape=s2)

        c = tf.matmul(a,b)
        print('{} @ {} = {}'.format(s1, s2, c.get_shape()))



    tf_matmul([2,3], [3,2])
    tf_matmul([3,2], [2,4])
    tf_matmul([5,3,2], [5,2,4])
    #   tf_matmul([5,3,2], [7,2,4]) #error

def show_squeeze():

    x = np.array([[[[[[0], [1], [2]], [[3], [4], [5]]]]]])
    print(x)
    print(x.shape)

    print(x.squeeze(axis=0).shape)
    print()

    #for axis in range(len(x.shape)):
    #    print(axis, np.squeeze(x, axis=axis).shape)

    print(np.squeeze(x))

def show_argmax():
    np.random.seed(12)
    a = np.random.choice(100, 12).reshape(3,4)
    print(a)
    print()
    print(np.argmax(a))
    print(np.argmax(a, axis=0))#수직(?,4) --> 3개의 수직 값 중 젤 높은 것
    print(np.argmax(a, axis=1))#수평(3,?)

    print('-' * 50)

    b = np.random.choice(100, 24).reshape(2,3,4)
    print(b)
    print()

    print(np.argmax(b, axis=0))#수직(?,3,4) --> 두개의 행렬 중 높은 것
    print(np.argmax(b, axis=1))#수평(2,?,4) --> 3개의 수직 값 중 젤 높은 것
    print(np.argmax(b, axis=2))#(2,3,?) --> 4개의 수평 값 중 젤 높은 것


show_matmul()
#show_squeeze()
show_argmax()

print (platform.architecture())
