import tensorflow as tf
import numpy as np

FLAGS = tf.app.flags.FLAGS

def _prepocess_data(x):
    transpose_x = tf.transpose(x, perm=[2, 0, 1])
    mean_val = tf.constant(np.loads(FLAGS.mean_file))
    x = tf.subtract(transpose_x, mean_val)
    x = tf.transpose(x, perm=[1, 2, 0])
    return tf.reshape(x, [FLAGS.image_size, FLAGS.image_size])



def _simple_conv(x, kernel, biases, strides, padding='SAME', active_fun=tf.nn.relu, name='conv'):
    conv1 = tf.nn.conv2d(x, kernel, strides, padding=padding, name=name) + biases
    conv1 = active_fun(conv1)
    return conv1


def _alexnet_head(x, base_decay=0.01, scope='alexnet'):
    with tf.name_scope(scope):
        tf.get_variable()
        conv1 = tf.layers.conv2d(x,
                                 filters=64,
                                 kernel_size=[11, 11],
                                 strides=4,
                                 padding='SAME',
                                 activation=tf.nn.relu,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                 kernel_regularizer=tf.contrib.layer.l2_regularizer(1*base_decay),
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
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                 bias_initializer=tf.ones_initializer(),
                                 kernel_regularizer=tf.contrib.layer.l2_regularizer(1*base_decay),
                                 name='conv2')
        pool2 = tf.layers.max_pooling2d(conv2, pool_size=[3, 3], strides=2, name='pool2')
        norm2 = tf.nn.lrn(pool2, alpha=1e-4, beta=0.75, name='norm2')

        conv3 = tf.layers.conv2d(norm2,
                                 filters=384,
                                 kernel_size=[3, 3],
                                 strides=1,
                                 padding='VALID',
                                 activation=tf.nn.relu,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                 kernel_regularizer=tf.contrib.layer.l2_regularizer(1*base_decay),
                                 name='conv3')
        conv4 = tf.layers.conv2d(conv3,
                                 filters=384,
                                 kernel_size=[3, 3],
                                 strides=1,
                                 padding='VALID',
                                 activation=tf.nn.relu,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                 bias_initializer=tf.ones_initializer(),
                                 kernel_regularizer=tf.contrib.layer.l2_regularizer(1*base_decay),
                                 name='conv4')

        conv5 = tf.layers.conv2d(conv4,
                                 filters=256,
                                 kernel_size=[3, 3],
                                 strides=1,
                                 padding='VALID',
                                 activation=tf.nn.relu,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                 bias_initializer=tf.ones_initializer(),
                                 kernel_regularizer=tf.contrib.layer.l2_regularizer(1*base_decay),
                                 name='conv5')
        pool5 = tf.layers.max_pooling2d(conv5, pool_size=[3, 3], strides=2, padding='SAME', name='pool5')

        fc6 = tf.layers.dense(pool5,
                              units=4096,
                              activation=tf.nn.relu,
                              kernel_initializer=tf.truncated_normal_initializer(stddev=0.005),
                              bias_initializer=tf.ones_initializer(),
                              kernel_regularizer=tf.contrib.layer.l2_regularizer(1*base_decay),
                              name='fc6')
        drop6 = tf.layers.dropout(fc6, rate=0.5, name="drop6")
        fc7 = tf.layers.dense(drop6,
                              units=4096,
                              activation=tf.nn.relu,
                              kernel_initializer=tf.truncated_normal_initializer(stddev=0.005),
                              bias_initializer=tf.ones_initializer(),
                              kernel_regularizer=tf.contrib.layer.l2_regularizer(1*base_decay),
                              name='fc7')
        drop7 = tf.layers.dropout(fc7, rate=0.5, name="drop6")

    return drop7

def ssdh_body(net,
              y,
              num_binary=30,
              num_class=6,
              base_decay=0.1,
              l1_weight=1,
              l2_weight=1,
              l3_weight=1):
    latent_sigmoid = tf.layers.dense(net, units=num_binary,
                    activation=tf.nn.sigmoid,
                    name='latent_sigmod')

    def _k1_euclidean_loss(name='loss1'):
        mean_vec = tf.constant(.5, dtype=tf.float32, shape=[num_binary])
        return tf.negative(tf.divide(tf.norm(tf.subtract(latent_sigmoid, mean_vec), tf.constant(num_binary))),
                           name=name)


    def _k2_euclidean_loss(name='loss2'):
        mean_val = tf.constant(.5, dtype=tf.float32, shape=[1])
        return tf.pow(tf.subtract(tf.reduce_sum(latent_sigmoid), mean_val), 2,
               name=name)

    def _classification_loss(name='loss'):
        fc9 = tf.layers.dense(latent_sigmoid,
                              units=num_class,
                              kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                              kernel_regularizer=tf.contrib.layer.l2_regularizer(1*base_decay),
                              name='fc9')
        return tf.nn.softmax_cross_entropy_with_logits(y, fc9, name=name)


    k1_loss = _k1_euclidean_loss(latent_sigmoid)
    k2_loss = _k2_euclidean_loss(latent_sigmoid)
    classification_loss = _classification_loss(latent_sigmoid)
    return tf.add(tf.add(tf.multiply(l1_weight, k1_loss),
                  tf.multiply(l2_weight, k2_loss)),
                  tf.multiply(l3_weight, classification_loss))


def ssdh_net():
    x = tf.placeholder(dtype=tf.float32, shape=[FLAGS.image_size, FLAGS.image_size, 3])
    y = tf.placeholder(dtype=tf.uint8, shape=[FLAGS.num_class])
    with tf.name_scope(name="ssdh"):
        x = _prepocess_data(x)
        net = _alexnet_head(x)
        net = ssdh_body(net, y, num_binary=30, num_class=6)
    return net

def ssdh_train(net):
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
    pass

def ssdh_evaluation(net):
    pass


if __name__ == "__main__":
    net = ssdh_net()
    ssdh_train(net)
    pass