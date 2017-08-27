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
import argparse
import cv2
import sys, os
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.saved_model import *

FLAGS = None

on = 0


def my_alignment(input):
    if on == 0:
        return input[0:input.shape[0] - 1]

    input_images = input[0:input.shape[0] - 1]
    w_image = input[input.shape[0] - 1]
    # org_shape = w_image.shape

    # imgs1 = np.array(input_images * 255, dtype=np.float32)
    # img2 = np.array(w_image * 255, dtype=np.float32)

    imgs1 = np.array(input_images, dtype=np.float32)
    img2 = np.array(w_image, dtype=np.float32)

    # stretch = 1
    # imgs1 = [cv2.resize(x, (stretch * x.shape[1], stretch * x.shape[0]), interpolation=cv2.INTER_CUBIC) for x in imgs1]
    # imgs1 = [cv2.threshold(x, 128, 255, cv2.THRESH_BINARY)[1] for x in imgs1]
    # imgs1 = [cv2.Laplacian(x, cv2.CV_8U) for x in imgs1]
    # img2 = cv2.resize(img2, (stretch * w_image.shape[1], stretch * w_image.shape[0]), interpolation=cv2.INTER_CUBIC)
    # img2 = cv2.threshold(img2, 128, 255, cv2.THRESH_BINARY)[1]
    # img2 = cv2.Laplacian(img2, cv2.CV_8U)

    # Find size of image1
    sz = imgs1[0].shape
    # Specify the number of iterations.
    # number_of_iterations = 5000;
    number_of_iterations = 50;

    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    # termination_eps = 1e-10;
    termination_eps = 0.001;

    # Define the motion model
    # warp_mode = cv2.MOTION_AFFINE
    warp_mode = cv2.MOTION_EUCLIDEAN
    # warp_mode = cv2.MOTION_HOMOGRAPHY

    transformedImages = []
    for i in range(0, len(input_images)):
        # Define 2x3 or 3x3 matrices and initialize the matrix to identity
        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            warp_matrix = np.eye(3, 3, dtype=np.float32)
        else:
            warp_matrix = np.eye(2, 3, dtype=np.float32)

        # Define termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

        try:
            # Run the ECC algorithm. The results are stored in warp_matrix.
            (cc, warp_matrix) = cv2.findTransformECC(img2, imgs1[i], warp_matrix, warp_mode, criteria)

            if warp_mode == cv2.MOTION_HOMOGRAPHY:
                # Use warpPerspective for Homography
                im1_aligned = cv2.warpPerspective(imgs1[i], warp_matrix, (sz[1], sz[0]),
                                                  flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
            else:
                # Use warpAffine for Translation, Euclidean and Affine
                im1_aligned = cv2.warpAffine(imgs1[i], warp_matrix, (sz[1], sz[0]),
                                             flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

                # im1_aligned = cv2.resize(im1_aligned, (org_shape[1],org_shape[0]), interpolation=cv2.INTER_CUBIC)
                # print("Done")
        except cv2.error:
            im1_aligned = imgs1[i]
            # im1_aligned = cv2.resize(im1_aligned, (org_shape[1], org_shape[0]), interpolation=cv2.INTER_CUBIC)
            # print("error while trying to find alignment")

        transformedImages.append(im1_aligned)
        # Show final results
        # cv2.imshow("Image 1", imgs1[i])
        # cv2.imshow("Image 2", img2)
        # cv2.imshow("Aligned Image 2", im2_aligned)
        # cv2.waitKey(0)
    transformedImages = np.reshape(transformedImages, [-1, 28, 28])
    # print(transformedImages.shape)
    # return transformedImages.astype(np.float32)
    return transformedImages
    # return input[0:input.shape[0]-1]


def my_get_wieghts(input_data, name=None):
    kernel = tf.Variable(tf.zeros([10, 784]))
    input_shaped = tf.reshape(input_data, [-1, 28, 28])

    res = []
    ws = []
    for k in range(kernel.shape[0].value):
        W = tf.reshape(kernel[k], [28, 28])

        # Rx = tf.py_func(my_alignment, [input_shaped, W], tf.float32)

        Rx = tf.py_func(my_alignment, [tf.concat([input_shaped, tf.reshape(W, [-1, 28, 28])], 0)], tf.float32)
        Rx = tf.stop_gradient(Rx)

        # stackedInputAndR = tf.stack([input_shaped, R])
        # transposedInputAndR = tf.transpose(stackedInputAndR, perm=[1,0,2,3])
        # Rx = tf.map_fn(lambda x: tf.matmul(x[1], x[0]), transposedInputAndR)
        Rx = tf.reshape(Rx, [-1, 784])
        W = tf.reshape(W, [784, 1])
        Y = tf.matmul(Rx, W)
        res.append(Y)

    return tf.reshape(tf.transpose(res), [-1, 10])


def main(_):
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    # tbd remove --------------------------------------
    # batch_xs, batch_ys = mnist.train.next_batch(100)
    # indexes = [np.array_equal(x,[1,0,0,0,0,0,0,0,0,0]) for x in batch_ys]
    # indexes = np.where(indexes)[0]
    # input_shaped = np.reshape(batch_xs[indexes], [-1, 28, 28])

    # temp_res = my_alignment2([input_shaped, np.reshape(batch_xs[90],[28, 28])])
    # temp_res = my_alignment2([input_shaped, input_shaped[1]])
    # --------------------------------

    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])

    #Wx = my_get_wieghts(x, name="proc_wieghts")
    W = tf.Variable(tf.zeros([784, 10]))
    Wx = tf.matmul(x, W)
    b = tf.Variable(tf.zeros([10]))
    y = Wx + b

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

    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    #builder = tf.saved_model_builder.SavedModelBuilder('Models\\Builder')
    saver = tf.train.Saver(max_to_keep=10)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # Train
    for  i in range(1000):
        if i == 50:
            global on
            on = 1
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch_xs, y_: batch_ys})
            print('step %d, training accuracy %g' % (i, train_accuracy))
            saver.save(sess, 'Models\model_{}.ckpt'.format(i))

    # Test trained model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("final")
    print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                        y_: mnist.test.labels}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
