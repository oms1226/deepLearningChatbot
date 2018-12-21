#Day_02_06_rnn_6.py
import numpy as np
import tensorflow as tf

from sklearn import preprocessing


def make_onshot_long_text(long_text, seq_length=20):
    data = list(''.join(long_text))
    lb = preprocessing.LabelBinarizer().fit(data)
    print(lb.classes_)

    onehot = lb.transform(data)

    x = np.float32(onehot[:-1])
    y = np.argmax(onehot[1:], axis=1)

    idx = [(i, i+seq_length) for i in range(len(onehot) - seq_length)]
    print(idx[:3], idx[-3:])#[(0, 20), (1, 21), (2, 22)] [(148, 168), (149, 169), (150, 170)]

    xx = [x[s:e] for s, e in idx]
    yy = [y[s:e] for s, e in idx]

    return np.float32(xx), tf.constant(np.int32(yy)), lb.classes_

def rnn_6_final(long_text):
    x, y, vocab = make_onshot_long_text(long_text)
    print(x.shape, y.shape, vocab)

    batch_size, seq_length, n_classes = x.shape
    hidden_size = 7

    text_len = [len(i) for i in long_text]

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
        print(i, sess.run(loss))
    print('-' * 50)

    pred = sess.run(z)
    pred_arg = np.argmax(pred, axis=2)

    total = '*' + ''.join(vocab[pred_arg[0]])
    print(total)

    for t in pred_arg[1:]:
        convert = ''.join(vocab[t])
        total += convert[-1]
        #print(convert)

    print(total)
    print(long_text)

    # print(i, pred_arg, vocab[pred_arg])
    # print(*vocab[sess.run(y)])# *는 unpacking 문법이다.
    sess.close()

#구글에서 drum up people 검색하여
#https://www.brainyquote.com/quotes/antoine_de_saintexupery_121261
text = ("If you want to build a ship, don't drum up people to collect wood "
        "and don't assign them tasks and work, but rather teach them "
        "to long for the endless immensity of the sea.")
rnn_6_final(text)


print('\n\n\n\n\n\n\n\n')