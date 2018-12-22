# Day_02_05_rnn_5_different.py
import numpy as np
import tensorflow as tf
from sklearn import preprocessing


# 문제
# 길이가 다른 여러 개의 문자열에 대해 동작하도록 수정하세요
def make_onehot_multi_different(text_array):
    data = list(''.join(text_array))
    lb = preprocessing.LabelBinarizer().fit(data)
    print(lb.classes_)

    max_len = max([len(i) for i in text_array])

    xx, yy = [], []
    for text in text_array:
        if len(text) < max_len:
            text += '*' * (max_len - len(text))

        onehot = lb.transform(list(text))

        x = onehot[:-1]
        y = np.argmax(onehot[1:], axis=1)

        xx.append(x)
        yy.append(list(y))

    return np.float32(xx), tf.constant(yy), lb.classes_


def rnn_5_different(text_array):
    x, y, vocab = make_onehot_multi_different(text_array)

    batch_size, seq_length, n_classes = x.shape
    hidden_size = 7

    text_len = [len(i) for i in text_array]

    cell = tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_size)
    outputs, _states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32,
                                         sequence_length=text_len)

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
        # print(i, pred_arg, vocab[pred_arg])
        # print(i, *[t[:3] for i, t in enumerate(vocab[pred_arg])])
        print(i, *[t[:text_len[j]-1] for j, t in enumerate(vocab[pred_arg])])

        # print(i, sess.run(loss))
    print('-' * 50)

    print(*vocab[sess.run(y)])
    sess.close()


rnn_5_different(['sky', 'coffee', 'rich'])




print('\n\n\n\n\n\n\n')
