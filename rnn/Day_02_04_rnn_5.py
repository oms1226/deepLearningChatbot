#Day_02_04_rnn_5.py
import numpy as np
import tensorflow as tf

from sklearn import preprocessing

def make_onshot_multi(text_array):
    #data = list(text_array[0] + text_array[1] + text_array[2])
    data = list(''.join(text_array))
    lb = preprocessing.LabelBinarizer().fit(data)
    print(lb.classes_)

    xx, yy = [], []
    for text in text_array:
        onehot = lb.transform(list(text))

        x = onehot[:-1]
        y = np.argmax(onehot[1:], axis=1)

        xx.append(x)
        yy.append(list(y))

    return np.float32(xx), tf.constant(yy), lb.classes_

def rnn_5(text_array):
    x, y, vocab = make_onshot_multi(text_array)
    print(x.shape, y.shape, vocab)

    batch_size, seq_length, n_classes = x.shape
    hidden_size = 7

    cell = tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_size)
    outputs, _states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)

    z = tf.layers.dense(outputs, n_classes)

    dummy_w = tf.ones([batch_size, seq_length])
    loss_i = tf.contrib.seq2seq.sequence_loss(logits=z, targets=y, weights=dummy_w)
    loss = tf.reduce_mean(loss_i)

    optimizer = tf.train.AdamOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(100):
        sess.run(train)

        pred = sess.run(z)
        pred_arg = np.argmax(pred, axis=2) #(1, 5, 6)
        # print(i, pred_arg, vocab[pred_arg])
        print(i, *vocab[pred_arg])

        # print(i, sess.run(loss))
    print('-' * 50)

    print(*vocab[sess.run(y)])
    sess.close()

rnn_5(['tensor', 'coffee', 'yellow'])


print('\n\n\n\n\n\n\n\n')