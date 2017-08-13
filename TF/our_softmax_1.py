# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import argparse
import sys
import numpy as np
from matplotlib import pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None

import os
import re
import sys
import tarfile
import scipy
import time

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from scipy import linalg as la


def matrix_symmetric(x):
    return (x + tf.transpose(x, [0, 2, 1])) / 2


def get_eigen_K(x, square=False):
    """
    Get K = 1 / (sigma_i - sigma_j) for i != j, 0 otherwise

    Parameters
    ----------
    x : tf.Tensor with shape as [..., dim,]

    Returns
    -------

    """
    if square:
        x = tf.square(x)
    res = tf.expand_dims(x, 1) - tf.expand_dims(x, 2)
    res += tf.eye(tf.shape(res)[1])
    res = 1 / res
    res -= tf.eye(tf.shape(res)[1])

    # Keep the results clean
    res = tf.where(tf.is_nan(res), tf.zeros_like(res), res)
    res = tf.where(tf.is_inf(res), tf.zeros_like(res), res)
    return res


def my_func(x):
    # x will be a numpy array with the contents of the placeholder below
    return np.sinh(x)


#@tf.RegisterGradient('Svd')
def gradient_svd(op, grad_s, grad_u, grad_v):
    diag_s = tf.map_fn(lambda x: tf.diag(x), grad_s)
    M = tf.matmul(grad_u, tf.matmul(diag_s, grad_v))
    #return M
    return tf.ones([100, 28, 28])

def get_procrustes_wieghts(input_data, kernel, name=None):
    input_shaped = tf.reshape(input_data, [-1, 28, 28])
    res = []
    Rmats = []
    for k in range(kernel.shape[1].value):
        W = tf.reshape(kernel[:, k], [28, 28])
        M = tf.map_fn(lambda x: tf.matmul(W, tf.transpose(x)), input_shaped)
        sM = tf.svd(M, full_matrices=True)
        R = tf.matmul(sM[1], sM[2], transpose_b=True)  # R = U * V^T
        RX = tf.reshape(tf.matmul(R, input_shaped), [-1, 28*28])  # R * X
        RX = tf.stop_gradient(RX)

        Y = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(RX, kernel[:, k])), axis=1))
        #Y = tf.reduce_sum(RX * kernel[:, k], axis=1)
        res.append(Y)
        Rmats.append(RX)

    Rmats = tf.transpose(tf.stack(Rmats), [1, 0, 2])
    input_kernel_similarities = tf.transpose(tf.stack(res))
    best_kernel_hits = tf.cast(tf.argmin(input_kernel_similarities, axis=1), tf.int32)
    best_Rs = tf.map_fn(lambda x: (x[0][x[1]], 1), (Rmats, best_kernel_hits))[0]
    rotated_inputs = tf.map_fn(lambda x: (x[0]*x[1], 1), (best_Rs, input_data))[0]
    output = tf.matmul(rotated_inputs, kernel)
    return output


def main(_):
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    # Create the model
    sess = tf.InteractiveSession()
    global_step_var = tf.train.get_or_create_global_step()

    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])
    b = tf.Variable(tf.zeros([10]))
    kernel = tf.Variable(tf.zeros([784, 10]))
    W = get_procrustes_wieghts(x, kernel, name="proc_wieghts")
    y0 = tf.matmul(x, kernel) + b
    y1 = W + b
    # Define loss and optimizer
    cross_entropy0 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y0))
    cross_entropy1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y1))
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    grads_and_vars0 = optimizer.compute_gradients(cross_entropy0)
    grads_and_vars1 = optimizer.compute_gradients(cross_entropy1)
    train0 = optimizer.apply_gradients(grads_and_vars0, global_step=global_step_var)
    train1 = optimizer.apply_gradients(grads_and_vars1, global_step=global_step_var)

    tf.global_variables_initializer().run()
    # Train
    for i in range(1000):
        if i < 30:  # burn-in
            train = train0
            y = y0
        else:
            train = train1
            y = y1
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})
        # Test trained model
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        if i % 10 == 0:
            print('iter ' + str(i) + ' - ' + str(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                            y_: mnist.test.labels})))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
