import tensorflow as tf
import numpy as np


FLAGS = tf.app.flags.FLAGS




def _alexnet_head(x, base_decay=FLAGS.weight_decay):

    conv1 = tf.layers.conv2d(x,
                             filters=96,
                             kernel_size=[11, 11],
                             strides=4,
                             padding='VALID',
                             activation=tf.nn.relu,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(1*base_decay),
                             name='conv1')
    pool1 = tf.layers.max_pooling2d(conv1,
                                    pool_size=[3, 3],
                                    strides=2,
                                    padding='VALID',
                                    name='pool1')
    norm1 = tf.nn.lrn(pool1,
                      depth_radius=5,
                      alpha=1e-4,
                      beta=0.75,
                      name='norm1')

    conv2 = tf.layers.conv2d(norm1,
                             filters=256,
                             kernel_size=[5, 5],
                             strides=1,
                             padding='SAME',
                             activation=tf.nn.relu,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                             bias_initializer=tf.ones_initializer(),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(1*base_decay),
                             name='conv2')
    pool2 = tf.layers.max_pooling2d(conv2,
                                    pool_size=[3, 3],
                                    strides=2,
                                    padding='VALID',
                                    name='pool2')
    norm2 = tf.nn.lrn(pool2,
                      depth_radius=5,
                      alpha=1e-4,
                      beta=0.75,
                      name='norm2')

    conv3 = tf.layers.conv2d(norm2,
                             filters=384,
                             kernel_size=[3, 3],
                             strides=1,
                             padding='SAME',
                             activation=tf.nn.relu,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(1*base_decay),
                             name='conv3')
    conv4 = tf.layers.conv2d(conv3,
                             filters=384,
                             kernel_size=[3, 3],
                             strides=1,
                             padding='SAME',
                             activation=tf.nn.relu,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                             bias_initializer=tf.ones_initializer(),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(1*base_decay),
                             name='conv4')

    conv5 = tf.layers.conv2d(conv4,
                             filters=256,
                             kernel_size=[3, 3],
                             strides=1,
                             padding='SAME',
                             activation=tf.nn.relu,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                             bias_initializer=tf.ones_initializer(),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(1*base_decay),
                             name='conv5')
    pool5 = tf.layers.max_pooling2d(conv5, pool_size=[3, 3], strides=2, padding='VALID', name='pool5')

    pool5_reshape = tf.reshape(pool5, [-1, pool5.shape[1].value * pool5.shape[2].value * pool5.shape[3].value])

    fc6 = tf.layers.dense(pool5_reshape,
                          units=4096,
                          activation=tf.nn.relu,
                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.005),
                          bias_initializer=tf.ones_initializer(),
                          kernel_regularizer=tf.contrib.layers.l2_regularizer(1*base_decay),
                          name='fc6')
    drop6 = tf.layers.dropout(fc6, rate=0.5)
    fc7 = tf.layers.dense(drop6,
                          units=4096,
                          activation=tf.nn.relu,
                          kernel_initializer=tf.truncated_normal_initializer(stddev=0.005),
                          bias_initializer=tf.ones_initializer(),
                          kernel_regularizer=tf.contrib.layers.l2_regularizer(1*base_decay),
                          name='fc7')
    drop7 = tf.layers.dropout(fc7, rate=0.5)

    return {"conv1": conv1, "pool1": pool1, "norm1": norm1,
            "conv2": conv2, "pool2": pool2, "norm2": norm2,
            "conv3": conv3, "conv4": conv4,
            "conv5": conv5, "pool5": pool5, "fc6": fc6, "fc7": fc7, "drop7": drop7}, drop7

def ssdh_body(net,
              tail,
              y,
              num_binary=30,
              num_class=6,
              base_decay=FLAGS.weight_decay,
              l1_weight=1,
              l2_weight=1,
              l3_weight=1):
    latent_sigmoid = tf.layers.dense(tail,
                                     units=num_binary,
                                     activation=tf.nn.sigmoid,
                                     kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(1 * base_decay),
                                     name='latent_sigmod')

    def _k1_euclidean_loss(name='loss1'):
        mean_vec = tf.constant(.5, dtype=tf.float32, shape=[num_binary])
        with tf.name_scope(name):
            return tf.reduce_mean(tf.square(tf.subtract(latent_sigmoid, mean_vec)))


    def _k2_euclidean_loss(name='loss2'):
        mean_val = tf.constant(.5, dtype=tf.float32)
        with tf.name_scope(name):
            return tf.pow(tf.subtract(tf.reduce_mean(latent_sigmoid), mean_val), 2)

    def _classification_loss(name='loss'):
        fc9 = tf.layers.dense(latent_sigmoid,
                              units=num_class,
                              kernel_initializer=tf.truncated_normal_initializer(stddev=0.2),
                              kernel_regularizer=tf.contrib.layers.l2_regularizer(1*base_decay),
                              name='fc9')

        return fc9, tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=fc9, name=name))


    k1_loss = _k1_euclidean_loss()
    k2_loss = _k2_euclidean_loss()
    fc9, classification_loss = _classification_loss()
    l1_weight = tf.constant(l1_weight, tf.float32)
    l2_weight = tf.constant(l2_weight, tf.float32)
    l3_weight = tf.constant(l3_weight, tf.float32)
    final_loss = tf.add(
                    tf.subtract(tf.multiply(l3_weight, classification_loss),
                                tf.multiply(l1_weight, k1_loss)),
                    tf.multiply(l2_weight, k2_loss))

    net['latent_sigmoid'] = latent_sigmoid
    net['k1_loss'] = k1_loss
    net['k2_loss'] = k2_loss
    net['classification_loss'] = classification_loss
    net['final_loss'] = final_loss
    net['fc9'] = fc9
    return net, final_loss


def ssdh_net(x, y):
    net, tail = _alexnet_head(x, base_decay=FLAGS.weight_decay)
    net, loss = ssdh_body(net,
                    tail,
                    y,
                    num_binary=FLAGS.num_binary,
                    num_class=FLAGS.num_class,
                    base_decay=FLAGS.weight_decay,
                    l1_weight=1,
                    l2_weight=1,
                    l3_weight=1)
    return net, loss


def load_fine_tune(sess):
    weights_dict = np.load(FLAGS.weight_path, encoding='bytes').item()
    weights_dict.pop("fc8")
    for op_name in weights_dict.keys():
        if op_name in ['conv2', 'conv4', 'conv5']:
            weights = np.append(weights_dict[op_name]['weights'], weights_dict[op_name]['weights'], 2)
        else:
            weights = weights_dict[op_name]['weights']
        biases = weights_dict[op_name]['biases']
        with tf.variable_scope(op_name, reuse=True):
            bias = tf.get_variable('bias', trainable=True)
            sess.run(bias.assign(biases))
            kernel = tf.get_variable('kernel', trainable=True)
            sess.run(kernel.assign(weights))



def get_train_op(loss, step):
    train_vals = tf.trainable_variables()
    grads = tf.gradients(loss, train_vals)

    kernels = train_vals[::2][:-1]
    biases = train_vals[1::2][:-1]
    final_kernel = train_vals[::2][-1:]
    final_biase = train_vals[1::2][-1:]

    g_kernels = grads[::2][:-1]
    g_biases = grads[1::2][:-1]
    g_final_kernel = grads[::2][-1:]
    g_final_biase = grads[1::2][-1:]

    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate,
                                               step,
                                               FLAGS.lr_decay_step,
                                               FLAGS.lr_decay,
                                               staircase=True)
    op1 = tf.train.MomentumOptimizer(learning_rate, FLAGS.momentum)
    op2 = tf.train.MomentumOptimizer(learning_rate * 2, FLAGS.momentum)
    op3 = tf.train.MomentumOptimizer(learning_rate * 10, FLAGS.momentum)
    op4 = tf.train.MomentumOptimizer(learning_rate * 20, FLAGS.momentum)

    train_op1 = op1.apply_gradients(zip(g_kernels, kernels))
    train_op2 = op2.apply_gradients(zip(g_biases, biases))
    train_op3 = op3.apply_gradients(zip(g_final_kernel, final_kernel))
    train_op4 = op4.apply_gradients(zip(g_final_biase, final_biase))
    train_op = tf.group(train_op1, train_op2, train_op3, train_op4)
    return train_op