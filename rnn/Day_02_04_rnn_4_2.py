#Day_02_04_rnn_4_2.py
import numpy as np
import tensorflow as tf

#scipy, scikit-Learn, sklearn 설치
from sklearn import preprocessing

def make_onshot_basic(text):
    idx2char = sorted(set(text))
    char2idx = {c: i for i, c in enumerate(idx2char)}

    print(idx2char)#['e', 'n', 'o', 'r', 's', 't']
    print(char2idx)#{'e': 0, 'n': 1, 'o': 2, 'r': 3, 's': 4, 't': 5}

    text_idx = [c for c in text]
    print(text_idx)
    text_idx = [char2idx[c] for c in text]
    print(text_idx)

    x = text_idx[:-1]
    y = text_idx[1:]
    print(x)
    print(y)

    eye = np.eye(len(char2idx))#단위행렬
    print(eye)
    print()

    xx = eye[x]#x 값을 2차원 행렬로 변경함
    print(xx)

    return np.float32([xx]), tf.constant([y]), np.array(idx2char)

def make_onshot(text):
    data = list(text)
    lb = preprocessing.LabelBinarizer().fit(data)
    print(lb)
    print(lb.classes_)

    onehot = lb.transform(data)
    print(onehot)

    x = onehot[:-1]
    y = np.argmax(onehot[1:], axis=1)

    #return np.float32([x]), tf.constant(y.reshape(-1, y.size)), lb.classes_
    return np.float32([x]), tf.constant(y.reshape(1, -1)), lb.classes_

def rnn_4(text):
    #x, y, vocab = make_onshot_basic(text)
    x, y, vocab = make_onshot(text)
    print(x.shape, y.shape, vocab)

    # # tensor
    # x = [[0., 0., 0., 0., 0., 1.],
    #      [1., 0., 0., 0., 0., 0.],
    #      [0., 1., 0., 0., 0., 0.],
    #      [0., 0., 0., 0., 1., 0.],
    #      [0., 0., 1., 0., 0., 0.]]
    # # ensor
    # y = [0, 1, 4, 2, 3]
    #
    # x = np.float32([x])

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
        print(i, pred_arg, vocab[pred_arg])

        # print(i, sess.run(loss))
    print('-' * 50)

    print(sess.run(y))
    sess.close()

rnn_4('tensor')
#rnn_4('deep_learning')


print('\n\n\n\n\n\n\n\n')