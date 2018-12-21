# Day_02_01_rnn_1.py
import numpy as np
import tensorflow as tf

# 'tensor' -> enorst
# input : tenso
# output: ensor

# 문제
# tensor 문자열을 원핫 레이블로 인코딩해서
# tenso가 ensor를 예측할 수 있도록 학습해보세요.
#
def rnn_1():
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

    w = tf.Variable(tf.random_uniform([6, 6]))
    b = tf.Variable(tf.random_uniform([6]))

    # (5, 6) = (5, 6) @ (6, 6)
    z = tf.matmul(x, w) + b
    hx = tf.nn.softmax(z)
    loss_i = tf.nn.softmax_cross_entropy_with_logits_v2(logits=z, labels=y)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(100):
        sess.run(train)

        pred = sess.run(hx)
        pred_arg = np.argmax(pred, axis=1)
        print(i, pred_arg, vocab[pred_arg])

        # print(i, sess.run(loss))
    print('-' * 50)

    print(np.argmax(y, axis=1))
    sess.close()


rnn_1()

print('\n\n\n\n\n\n\n')