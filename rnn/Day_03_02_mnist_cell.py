#Day_03_02_mnist_cell.py

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('mnist', one_hot=True)

print(mnist.train.images.shape) #(55000, 784)
print(mnist.train.labels.shape) #(55000, 10)

print(mnist.validation.images.shape) #(5000, 784)
print(mnist.test.images.shape) #(10000, 784)