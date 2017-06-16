import tensorflow as tf
import numpy as np

FLAGS = tf.app.flags.FLAGS

def _prepocess_data(x):
    transpose_x = tf.transpose(x, perm=[2, 0, 1])
    mean_val = tf.constant(np.loads(FLAGS.mean_file))
    transpose_x -= mean_val
    x = tf.transpose(x, perm=[1, 2, 0])
    return tf.reshape(x, [FLAGS.image_size, FLAGS.image_size])



def _simple_conv(x, kernel, biases, strides, padding='SAME', active_fun=tf.nn.relu, name='conv'):
    conv1 = tf.nn.conv2d(x, kernel, strides, padding=padding, name=name) + biases
    conv1 = active_fun(conv1)
    return conv1

def _alexnet_head(x, scope='alexnet'):
    with tf.name_scope(scope):
        conv1 = tf.layers.conv2d(x,
                                 filters=64,
                                 kernel_size=[11, 11],
                                 strides=4,
                                 padding='SAME',
                                 activation=tf.nn.relu,
                                 name='conv1')
        pool1 = tf.layers.max_pooling2d(conv1,
                                        pool_size=[3, 3],
                                        strides=2,
                                        name='pool1')
        norm1 = tf.nn.lrn(pool1,
                          alpha=1e-4,
                          beta=0.75,
                          name='norm1')

        conv2 = tf.layers.conv2d(norm1,
                                 filters=256,
                                 kernel_size=[5, 5],
                                 strides=1,
                                 padding='VALID',
                                 activation=tf.nn.relu,
                                 name='conv2')
        pool2 = tf.layers.max_pooling2d(conv2, pool_size=[3, 3], strides=2, name='pool2')
        norm2 = tf.nn.lrn(pool2, alpha=1e-4, beta=0.75, name='norm2')

        conv3 = tf.layers.conv2d(norm2,
                                 filters=384,
                                 kernel_size=[3, 3],
                                 strides=1,
                                 padding='VALID',
                                 activation=tf.nn.relu,
                                 name='conv3')
        conv4 = tf.layers.conv2d(conv3,
                                 filters=384,
                                 kernel_size=[3, 3],
                                 strides=1,
                                 padding='VALID',
                                 activation=tf.nn.relu,
                                 name='conv4')

        conv5 = tf.layers.conv2d(conv4,
                                 filters=256,
                                 kernel_size=[3, 3],
                                 strides=1,
                                 padding='VALID',
                                 activation=tf.nn.relu,
                                 name='conv5')
        pool5 = tf.layers.max_pooling2d(conv5, pool_size=[3, 3], strides=2, padding='SAME', name='pool5')

        fc6 = tf.layers.dense(pool5, units=4096, activation=tf.nn.relu, name='fc6')

        fc7 = tf.layers.dense(fc6, units=4096, activation=tf.nn.relu, name='fc7')

    return fc7

def ssdh_net(x):
    with tf.name_scope(name="ssdh") as ns:
        x = _prepocess_data(x)
        net = _alexnet_head()

    pass

def ssdh_train(net, X, Y):
    pass

def ssdj_evaluation(net, X, Y):
    pass
