import numpy as np
import tensorflow as tf



def rnn_3_1():
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

    batch_size, seq_length, n_classes = x.shape
    hidden_size = 7

    cell = tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_size)
    outputs, _states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
    print(outputs.get_shape())    # (1, 5, 2)

    w = tf.Variable(tf.random_uniform([hidden_size, n_classes]))
    b = tf.Variable(tf.random_uniform([n_classes]))

    # (5, 6) = (5, 2) @ (2, 6)
    z = tf.matmul(outputs[0], w) + b
    # hx = tf.nn.softmax(z)

    #------------------------------#

    #loss_i = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=z, labels=y)

    #y = [y]#tf.constant([y])
    y = tf.constant([y])
    z = tf.reshape(z, [batch_size, seq_length, n_classes])
    #3차원을 다루어야 되기에 위에를 안쓴다.
    dummy_w = tf.ones([batch_size, seq_length])
    loss_i = tf.contrib.seq2seq.sequence_loss(logits=z, targets=y, weights=dummy_w)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(100):
        sess.run(train)

        pred = sess.run(z)
        pred_arg = np.argmax(pred, axis=2) #(1, 5, 6)
        print(i, pred_arg, vocab[pred_arg])

        # print(i, sess.run(loss))
    print('-' * 50)

    print(sess.run(y))
    sess.close()


def rnn_3_2():
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

    batch_size, seq_length, n_classes = x.shape
    hidden_size = 7

    cell = tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_size)
    outputs, _states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
    print(outputs.get_shape())    # (1, 5, 2)

    w = tf.Variable(tf.random_uniform([hidden_size, n_classes]))
    b = tf.Variable(tf.random_uniform([n_classes]))

    # (1, 5, 6) = (1, 5, 7) @ (1, 7, 6)
    #z = tf.matmul(outputs[0], [w]) + b

    # z = tf.contrib.layers.fully_connected(inputs=outputs,
    #                                       num_outputs=n_classes,
    #                                       activation_fn=None,
    #                                       weights_initializer=tf.zeros_initializer())

    z = tf.layers.dense(outputs, n_classes)

    #------------------------------#
    y = tf.constant([y])
    #z = tf.reshape(z, [batch_size, seq_length, n_classes])
    #3차원을 다루어야 되기에 위에를 안쓴다.
    dummy_w = tf.ones([batch_size, seq_length])
    loss_i = tf.contrib.seq2seq.sequence_loss(logits=z, targets=y, weights=dummy_w)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(100):
        sess.run(train)

        pred = sess.run(z)
        pred_arg = np.argmax(pred, axis=2) #(1, 5, 6)
        print(i, pred_arg, vocab[pred_arg])

        # print(i, sess.run(loss))
    print('-' * 50)

    print(sess.run(y))
    sess.close()

#rnn_3_1()
rnn_3_2()

print('\n\n\n\n\n\n\n')