# Day_02_03_rnn_3_3.py
import numpy as np
import tensorflow as tf


def show_sequence_loss(targets, logits):
    y = tf.constant(targets)
    z = tf.constant(logits)

    w = tf.ones([1, len(targets[0])])

    loss = tf.contrib.seq2seq.sequence_loss(logits=z, targets=y, weights=w)

    with tf.Session():
        print(loss.eval(), targets, logits)


pred1 = [[[0.2, 0.7], [0.6, 0.2], [0.2, 0.9]]]
pred2 = [[[0.7, 0.2], [0.2, 0.6], [0.9, 0.2]]]

show_sequence_loss([[1, 1, 1]], pred1)
show_sequence_loss([[0, 0, 0]], pred2)

# show_sequence_loss([[0.5, 0.5, 0.5]], pred1)      # error
# show_sequence_loss([[2, 2, 2]], pred1)            # error
# show_sequence_loss([[1, 1, 1, 1]], pred1)         # error

show_sequence_loss([[2, 2, 2]], [[[0.2, 0.7, 0.5], [0.6, 0.2, 0.5], [0.2, 0.9, 0.1]]])
show_sequence_loss([[1, 1, 1]], [[[0.2, 0.7, 0.5], [0.6, 0.2, 0.5], [0.2, 0.9, 0.1]]])

show_sequence_loss([[1, 1, 1, 1]], [[[0.2, 0.7], [0.2, 0.7], [0.6, 0.2], [0.2, 0.9]]])
