# Day_02_04_rnn_4_3_word.py
import numpy as np
import tensorflow as tf
from sklearn import preprocessing


def make_onehot(text):
    data = list(text)
    lb = preprocessing.LabelBinarizer().fit(data)

    onehot = lb.transform(data)

    x = onehot[:-1]
    y = np.argmax(onehot[1:], axis=1)

    return np.float32([x]), tf.constant(y.reshape(1, -1)), lb.classes_


def word_rnn(words):
    x, y, vocab = make_onehot(words)

    batch_size, seq_length, n_classes = x.shape
    hidden_size = 7

    cell = tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_size)
    outputs, _states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)

    z = tf.layers.dense(outputs, n_classes)

    dummy_w = tf.ones([batch_size, seq_length])
    loss = tf.contrib.seq2seq.sequence_loss(logits=z, targets=y, weights=dummy_w)

    optimizer = tf.train.AdamOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(100):
        sess.run(train)

        pred = sess.run(z)
        pred_arg = np.argmax(pred, axis=2)      # (1, 5, 6)
        print(i, pred_arg, vocab[pred_arg])

        # print(i, sess.run(loss))
    print('-' * 50)

    print(sess.run(y))
    sess.close()


#           t       e      n      s        o       r
numbers = ['zero', 'one', 'two', 'three', 'four', 'five']

word_rnn(numbers)
