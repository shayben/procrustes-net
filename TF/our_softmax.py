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
import cv2
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

def my_func(input):
    MIN_MATCH_COUNT = 10

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    input_images = np.reshape(input[0], [-1,28,28])
    w_image = input[1]
    cv_w_image = np.array(w_image * 255, dtype=np.uint8)
    cv_input_images = np.array(input_images * 255, dtype=np.uint8)

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(cv_w_image, None)
    res2 = [sift.detectAndCompute(x, None) for x in cv_input_images]

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = [bf.knnMatch(des1, x[1],k=2) for x in res2]

    good = []
    for i in range(0, input_images.shape[0]):
        for m, n in matches[i]:
            if m.distance < 0.7 * n.distance:
                good[i].append(m)

    for i in range(0, input_images.shape[0]):
        if len(good[i]) > MIN_MATCH_COUNT:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good[i]]).reshape(-1, 1, 2)
            dst_pts = np.float32([res2[i][0][m.trainIdx].pt for m in good[i]]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()
            h, w = w_image.shape

            #tbd finish -------------------------------
            #pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            #dst = cv2.perspectiveTransform(pts, M)
            #transfomedImage = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)


@tf.RegisterGradient('Svd')
def gradient_svd(op, grad_s, grad_u, grad_v):
    return tf.zeros([100,28,28])
    # return dxdz

def get_opencv_wieghts(input_data, name=None):
    kernel = tf.Variable(tf.zeros([10,784]))
    input_shaped = tf.reshape(input_data, [-1, 28,28])

    res = []

    for k in range(kernel.shape[0].value):
        W = tf.reshape(kernel[k],[28,28])
        R = tf.py_func(my_func, [input_shaped, W], tf.float32)
        stackedInputAndR = tf.stack([input_shaped, R])
        transposedInputAndR = tf.transpose(stackedInputAndR, perm=[1,0,2,3])
        Rx = tf.map_fn(lambda x: tf.matmul(x[1], x[0]), transposedInputAndR)
        Rx = tf.reshape(Rx, [-1, 784])
        W = tf.reshape(W, [784,1])
        Y = tf.matmul(Rx, W)
        res.append(Y)

    return tf.reshape(res, [-1, 10])

def main(_):
  # Import data
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  # tbd remove --------------------------------------
  batch_xs, batch_ys = mnist.train.next_batch(10)
  input_shaped = np.reshape(batch_xs, [-1, 28, 28])
  W = np.random.random((28, 28))
  my_func([input_shaped, W])
  #--------------------------------------

  #tbd calculate gradient

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])

  W = get_opencv_wieghts(x, name="proc_wieghts")
  b = tf.Variable(tf.zeros([10]))
  y = W + b

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])

  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
  # outputs of 'y', and then average across the batch.
  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()
  # Train
  for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

  # Test trained model
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                      y_: mnist.test.labels}))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)