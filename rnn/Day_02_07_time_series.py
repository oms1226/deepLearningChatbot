#Day_02_07_time_series.py
import numpy as np
import tensorflow as tf

from sklearn import preprocessing, model_selection
import matplotlib.pyplot as plt

def rnn_stock_1():
    def minmax_scale(data):
        mn = np.min(data, 0)
        mx = np.max(data)
        return (data - mn) / (mx - mn)


    seq_length = 7 #7일분의 data
    hidden_size = 10 #마음대로 조정 가능하다
    n_classes = 5 #feature들 --> Open,High,Low,Volume,Close
    output_dim = 1 #예측할 갯수

    stock = np.loadtxt('Data/stock_daily.csv', delimiter=',')
    print(stock.shape) #(732, 5)

    stock = stock[::-1]#주식 데이터는 날짜 최신 순으로 되어 있기에 과거순부터 정리하도록 역순으로 바꾼다.
    # tf.random_shuffle(stock) 시계열 데이터를 셔플하면 안된다!!!!!
    stock = minmax_scale(stock)#정규화

    x = stock
    #y = stock[:, -1]#(732,)#차원이 떨어진다.
    y = stock[:, -1:]#(732, 1)#차원을 높인다.
    print(y.shape)

    xx, yy = [], []
    for i in range(len(y) - seq_length):
        xx.append(x[i:i+seq_length])
        yy.append(y[i+seq_length])#값 하나만 예측하기에

    print(len(xx))

    xx = np.float32(xx)
    yy = np.float32(yy)

    cell = tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_size)
    outputs, _states = tf.nn.dynamic_rnn(cell, xx, dtype=tf.float32)

    z = tf.layers.dense(outputs[:, -1], output_dim)
    loss = tf.reduce_mean((z - yy) ** 2)

    optimizer = tf.train.AdamOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(100):
        sess.run(train)
        print(i, sess.run(loss))

    print('-' * 50)
    sess.close()


# 문제
# 데이터를 7:3으로 나누어서 학습하고 검사하세요
def rnn_stock_2():
    seq_length = 7 #7일분의 data
    hidden_size = 10 #마음대로 조정 가능하다
    n_classes = 5 #feature들 --> Open,High,Low,Volume,Close
    output_dim = 1 #예측할 갯수

    stock = np.loadtxt('Data/stock_daily.csv', delimiter=',')
    print(stock.shape) #(732, 5)

    stock = stock[::-1]#주식 데이터는 날짜 최신 순으로 되어 있기에 과거순부터 정리하도록 역순으로 바꾼다.
    stock = preprocessing.minmax_scale(stock)

    x = stock
    #y = stock[:, -1]#(732,)#차원이 떨어진다.
    y = stock[:, -1:]#(732, 1)#차원을 높인다.
    print(y.shape)

    xx, yy = [], []
    for i in range(len(y) - seq_length):
        xx.append(x[i:i+seq_length])
        yy.append(y[i+seq_length])#값 하나만 예측하기에

    print(len(xx))

    xx = np.float32(xx)
    yy = np.float32(yy)

    train_size = int(len(xx) * 0.7)
    x_train, x_test = xx[:train_size], xx[train_size:]
    y_train, y_test = yy[:train_size], yy[train_size:]

    ph_x = tf.placeholder(tf.float32, [None, seq_length, n_classes])

    cell = tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_size)
    outputs, _states = tf.nn.dynamic_rnn(cell, ph_x, dtype=tf.float32)

    z = tf.layers.dense(outputs[:, -1], output_dim)
    loss = tf.reduce_mean((z - y_train) ** 2)

    optimizer = tf.train.AdamOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(100):
        sess.run(train, {ph_x: x_train})
        print(i, sess.run(loss, {ph_x: x_train}))

    print('-' * 50)

    pred = sess.run(z, {ph_x: x_test})#
    pred = np.squeeze(pred)
    print(pred)
    sess.close()

    plt.plot(y_test, 'r')
    plt.plot(pred, 'g')
    plt.show()


def rnn_stock_3():
    seq_length = 7 #7일분의 data
    hidden_size = 10 #마음대로 조정 가능하다
    n_classes = 5 #feature들 --> Open,High,Low,Volume,Close
    output_dim = 1 #예측할 갯수

    stock = np.loadtxt('Data/stock_daily.csv', delimiter=',')
    print(stock.shape) #(732, 5)

    stock = stock[::-1]#주식 데이터는 날짜 최신 순으로 되어 있기에 과거순부터 정리하도록 역순으로 바꾼다.
    stock = preprocessing.minmax_scale(stock)

    x = stock
    #y = stock[:, -1]#(732,)#차원이 떨어진다.
    y = stock[:, -1:]#(732, 1)#차원을 높인다.
    print(y.shape)

    xx, yy = [], []
    for i in range(len(y) - seq_length):
        xx.append(x[i:i+seq_length])
        yy.append(y[i+seq_length])#값 하나만 예측하기에

    print(len(xx))

    xx = np.float32(xx)
    yy = np.float32(yy)

    train_size = int(len(xx) * 0.7)
    x_train, x_test = xx[:train_size], xx[train_size:]
    y_train, y_test = yy[:train_size], yy[train_size:]

    ph_x = tf.placeholder(tf.float32, [None, seq_length, n_classes])

    cell = tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_size)
    outputs, _states = tf.nn.dynamic_rnn(cell, ph_x, dtype=tf.float32)

    z = tf.layers.dense(outputs[:, -1], output_dim)
    loss = tf.reduce_mean((z - y_train) ** 2)

    optimizer = tf.train.AdamOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(100):
        sess.run(train, {ph_x: x_train})
        print(i, sess.run(loss, {ph_x: x_train}))

    print('-' * 50)

    saver = tf.train.Saver()
    saver.save(sess, 'model/stock')

    # pred = sess.run(z, {ph_x: x_test})#
    # pred = np.squeeze(pred)
    # print(pred)
    # sess.close()
    #
    # plt.plot(y_test, 'r')
    # plt.plot(pred, 'g')
    # plt.show()

def rnn_stock_4():
    seq_length = 7 #7일분의 data
    hidden_size = 10 #마음대로 조정 가능하다
    n_classes = 5 #feature들 --> Open,High,Low,Volume,Close
    output_dim = 1 #예측할 갯수

    stock = np.loadtxt('Data/stock_daily.csv', delimiter=',')
    print(stock.shape) #(732, 5)

    stock = stock[::-1]#주식 데이터는 날짜 최신 순으로 되어 있기에 과거순부터 정리하도록 역순으로 바꾼다.
    stock = preprocessing.minmax_scale(stock)

    x = stock
    #y = stock[:, -1]#(732,)#차원이 떨어진다.
    y = stock[:, -1:]#(732, 1)#차원을 높인다.
    print(y.shape)

    xx, yy = [], []
    for i in range(len(y) - seq_length):
        xx.append(x[i:i+seq_length])
        yy.append(y[i+seq_length])#값 하나만 예측하기에

    print(len(xx))

    xx = np.float32(xx)
    yy = np.float32(yy)

    train_size = int(len(xx) * 0.7)
    x_train, x_test = xx[:train_size], xx[train_size:]
    y_train, y_test = yy[:train_size], yy[train_size:]

    ph_x = tf.placeholder(tf.float32, [None, seq_length, n_classes])

    cell = tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_size)
    outputs, _states = tf.nn.dynamic_rnn(cell, ph_x, dtype=tf.float32)

    z = tf.layers.dense(outputs[:, -1], output_dim)
    loss = tf.reduce_mean((z - y_train) ** 2)

    optimizer = tf.train.AdamOptimizer(0.1)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    # sess.run(tf.global_variables_initializer())

    # for i in range(100):
    #     sess.run(train, {ph_x: x_train})
    #     print(i, sess.run(loss, {ph_x: x_train}))
    #
    # print('-' * 50)

    latest = tf.train.latest_checkpoint('model')
    print(latest)

    saver = tf.train.Saver()
    #saver.restore(sess, 'model/stock')
    saver.restore(sess, latest)#latest가 'model/stock'으로 나오기에 동일하나 latest 가 바뀌면 다르다.

    pred = sess.run(z, {ph_x: x_test})#
    pred = np.squeeze(pred)
    print(pred)
    sess.close()

    plt.plot(y_test, 'r')
    plt.plot(pred, 'g')
    plt.show()

#rnn_stock_1()
#rnn_stock_2()#학습한 데이터를 사용하여 test 데이터를 예측한다.
#rnn_stock_3()#학습 모델을 저장한다.
rnn_stock_4()#학습 모델을 사용한다.
