import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

#meta_graph_file = r'c:\tmp\cifar10_train_procrustes\model.ckpt-81587.meta'
#model_checkpoint_file = r'c:\tmp\cifar10_train_procrustes\model.ckpt-81587'

meta_graph_file = r'c:\tmp\cifar10_train_procrustes2\model.ckpt-10000.meta'
model_checkpoint_file = r'c:\tmp\cifar10_train_procrustes2\model.ckpt-10000'

with tf.Session() as sess:
    saver = tf.train.import_meta_graph(meta_graph_file)
    saver.restore(sess, model_checkpoint_file)
    variables = tf.global_variables()
    procrustes_layer = variables[1]

    for i in range(64):
        kernel = np.squeeze(procrustes_layer[:, :, :, i].eval())
        plt.subplot(8, 8, i+1)
        plt.imshow(kernel, cmap=plt.cm.BuPu_r)

    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    cax = plt.axes([0.85, 0.1, 0.075, 0.8])
    plt.colorbar(cax=cax)
    plt.show()

print('Done.')