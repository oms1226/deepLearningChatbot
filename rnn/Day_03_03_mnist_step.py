#Day_03_03_mnist_step.py

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data



def mnist_rnn_step():
    mnist = input_data.read_data_sets('mnist', one_hot=True)

    print(mnist.train.images.shape)  # (55000, 784)
    print(mnist.train.labels.shape)  # (55000, 10)

    print(mnist.validation.images.shape)  # (5000, 784)
    print(mnist.test.images.shape)  # (10000, 784)

    elem_size = 28#수평
    time_step = 28#수직 #sequence_length
    n_classes = 10
    batch_size = 100    #128
    hidden_size = 150

    #ph_x = tf.placeholder(tf.float32, [None, elem_size*time_step])#ValueError: Shape (784, ?) must have rank at least 3
    ph_x = tf.placeholder(tf.float32, [None, time_step, elem_size])#위에 에러로 아래와 같이 수정
    ph_y = tf.placeholder(tf.float32, [None, n_classes])

    ####수정전#######
    # cell = tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_size)
    # outputs, _states = tf.nn.dynamic_rnn(cell, ph_x, dtype=tf.float32)
    # final = outputs[:, -1, :] # --> (100, 150)
    # z = tf.layers.dense(final, n_classes)
    ####수정후(cell 내부를 직접 구현 nlp_rnn.pdf 138page를 직접 구현)#######
    w_h = tf.Variable(tf.zeros([hidden_size, hidden_size])) #hidden state
    w_x = tf.Variable(tf.zeros([elem_size, hidden_size]))  # inputs
    b_x = tf.Variable(tf.zeros([hidden_size]))  # inputs

    def rnn_step(prev_hidden_state, x):
        # (100, 150) = (100, 150) @ (150, 150)
        prev = tf.matmul(prev_hidden_state, w_h)

        # (100, 150) = (100, 28) @ (28, 150)
        # (batch_size, hidden_size) = (batch_size, elem_size) * (elem_size, hidden_size)
        curr = tf.matmul(x, w_x) + b_x

        # (100, 150) = (100, 150) + (100, 150)
        curr_hidden_state = tf.tanh(prev + curr)
        return curr_hidden_state

    #(100, 28, 28) 인데 맨 앞쪽에 숫자가 반복하는 횟수이기에 28번 전달해야 되기에
    # --> (28, 100, 28) 아래와 같은 transpose 작업을 한다.
    preocessed_inputs = tf.transpose(ph_x, perm=[1,0,2])

    initial_hidden = tf.zeros([batch_size, hidden_size])
    outputs = tf.scan(rnn_step, preocessed_inputs, initializer=initial_hidden)
    #rnn_step 안에 prev_hidden_state 값의 초기 값을 initial_hidden로 정한 것이다.

    w_l = tf.Variable(tf.truncated_normal([hidden_size, n_classes], mean=0, stddev=0.01))
    b_l = tf.Variable(tf.truncated_normal([n_classes], mean=0, stddev=0.01))

    #z = tf.matmul(outputs[-1], w_l) + b_l
    #위와 같이 해도 동작한다.
    def get_linear(prev_outputs):
        return tf.matmul(prev_outputs, w_l) + b_l

    z = tf.map_fn(get_linear, outputs)
    print(z.shape)#(28, 100, 10)

    z = z[-1]
    print(z.shape)#(100, 10)
    #######################
    loss_i = tf.nn.softmax_cross_entropy_with_logits_v2(logits=z, labels=ph_y)
    loss = tf.reduce_mean(loss_i)

    #optimizer = tf.train.GradientDescentOptimizer(0.1)
    optimizer = tf.train.RMSPropOptimizer(0.001)
    train = optimizer.minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(1000):
        xx, yy = mnist.train.next_batch(batch_size)  # 100개만큼 데이터를 가져오고, 다 가져오면 셔플해서 가져온다.
        xx = xx.reshape(batch_size, time_step, elem_size) # (100, 28, 28)
        sess.run(train, {ph_x: xx, ph_y: yy})

        if i % 100 == 0:
            print(i, sess.run(loss, {ph_x: xx, ph_y: yy}))
    print('-' * 50)

    # pred = sess.run(hx, {ph_x: x})
    pred = sess.run(z, {ph_x: mnist.test.images[:batch_size].reshape(-1, time_step, elem_size)})
    pred_arg = np.argmax(pred, axis=1)
    y_arg = np.argmax(mnist.test.labels[:batch_size], axis=1)

    equals = ()
    print(pred_arg)
    print(y_arg)

    equals = (pred_arg == y_arg)
    print('acc :', np.mean(equals))
    print('-' * 50)

    sess.close()


mnist_rnn_step()

print('\n\n\n\n\n')