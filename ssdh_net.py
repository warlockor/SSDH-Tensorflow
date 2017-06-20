from __future__ import absolute_import
import tensorflow as tf
import numpy as np
import os
import random

tf.app.flags.DEFINE_integer("dim_size", 256, "image size into netword")

tf.app.flags.DEFINE_integer("image_size", 227, "image size into netword")
tf.app.flags.DEFINE_string("mean_file", "./data/ilsvrc_2012_mean.npy", "mean file")
tf.app.flags.DEFINE_integer("num_class", 3, "classification number")
tf.app.flags.DEFINE_integer("num_binary", 800, "number of binary code")
tf.app.flags.DEFINE_integer("weight_decay", 0.0005, "l2 weight regularization decay")
tf.app.flags.DEFINE_string("train_dir", "./data/train", "train data dir")
tf.app.flags.DEFINE_string("test_dir", "./data/test", "train data dir")

tf.app.flags.DEFINE_bool("is_train", True, "train or test")
tf.app.flags.DEFINE_integer("train_batch_size", 32, "batch size in train")
tf.app.flags.DEFINE_integer("test_batch_size", 50, "batch size in test")

tf.app.flags.DEFINE_float("learning_rate", 0.001, "learning rate")
tf.app.flags.DEFINE_integer("max_iter", 50000, "max iterator number")
tf.app.flags.DEFINE_integer("print_error_step", 100, 'print_error_step')
tf.app.flags.DEFINE_integer("eval_step", 100, 'print_error_step')

tf.app.flags.DEFINE_integer("lr_decay_step", 10000, 'learning decay step')
tf.app.flags.DEFINE_integer("lr_decay", 0.1, 'learning decay step')

FLAGS = tf.app.flags.FLAGS




def _alexnet_head(x, base_decay=0.01, scope='alexnet'):
    with tf.name_scope(scope):
        conv1 = tf.layers.conv2d(x,
                                 filters=64,
                                 kernel_size=[11, 11],
                                 strides=4,
                                 padding='SAME',
                                 activation=tf.nn.relu,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(1*base_decay),
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
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(1*base_decay),
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
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(1*base_decay),
                                 name='conv3')
        conv4 = tf.layers.conv2d(conv3,
                                 filters=384,
                                 kernel_size=[3, 3],
                                 strides=1,
                                 padding='VALID',
                                 activation=tf.nn.relu,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                 bias_initializer=tf.ones_initializer(),
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(1*base_decay),
                                 name='conv4')

        conv5 = tf.layers.conv2d(conv4,
                                 filters=256,
                                 kernel_size=[3, 3],
                                 strides=1,
                                 padding='VALID',
                                 activation=tf.nn.relu,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                 bias_initializer=tf.ones_initializer(),
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(1*base_decay),
                                 name='conv5')
        pool5 = tf.layers.max_pooling2d(conv5, pool_size=[3, 3], strides=2, padding='SAME', name='pool5')

        pool5_reshape = tf.reshape(pool5, [-1, pool5.shape[1]._value * pool5.shape[2]._value * pool5.shape[3]._value])

        fc6 = tf.layers.dense(pool5_reshape,
                              units=4096,
                              activation=tf.nn.relu,
                              kernel_initializer=tf.truncated_normal_initializer(stddev=0.005),
                              bias_initializer=tf.ones_initializer(),
                              kernel_regularizer=tf.contrib.layers.l2_regularizer(1*base_decay),
                              name='fc6')
        drop6 = tf.layers.dropout(fc6, rate=0.5, name="drop6")
        fc7 = tf.layers.dense(drop6,
                              units=4096,
                              activation=tf.nn.relu,
                              kernel_initializer=tf.truncated_normal_initializer(stddev=0.005),
                              bias_initializer=tf.ones_initializer(),
                              kernel_regularizer=tf.contrib.layers.l2_regularizer(1*base_decay),
                              name='fc7')
        drop7 = tf.layers.dropout(fc7, rate=0.5, name="drop7")

    return {"conv1": conv1, "pool1": pool1, "norm1": norm1,
            "conv2": conv2, "pool2": pool2, "norm2": norm2,
            "conv3": conv3, "conv4": conv4,
            "conv5": conv5, "pool5": pool5, "fc6": fc6, "fc7": fc7, "drop7": drop7}, drop7

def ssdh_body(net,
              tail,
              y,
              num_binary=30,
              num_class=6,
              base_decay=0.1,
              l1_weight=1,
              l2_weight=1,
              l3_weight=1):
    latent_sigmoid = tf.layers.dense(tail, units=num_binary,
                    activation=tf.nn.sigmoid,
                    name='latent_sigmod')

    def _k1_euclidean_loss(name='loss1'):
        mean_vec = tf.constant(.5, dtype=tf.float32, shape=[num_binary])
        with tf.name_scope(name):
            return tf.negative(tf.divide(tf.norm(tf.subtract(latent_sigmoid, mean_vec)), tf.constant(num_binary, tf.float32)))


    def _k2_euclidean_loss(name='loss2'):
        mean_val = tf.constant(.5, dtype=tf.float32, shape=[1])
        with tf.name_scope(name):
            return tf.pow(tf.subtract(tf.reduce_sum(latent_sigmoid), mean_val), 2)

    def _classification_loss(name='loss'):
        fc9 = tf.layers.dense(latent_sigmoid,
                              units=num_class,
                              kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                              kernel_regularizer=tf.contrib.layers.l2_regularizer(1*base_decay),
                              name='fc9')

        return fc9, tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=fc9, name=name))


    k1_loss = _k1_euclidean_loss()
    k2_loss = _k2_euclidean_loss()
    fc9, classification_loss = _classification_loss()
    l1_weight = tf.constant(l1_weight, tf.float32)
    l2_weight = tf.constant(l2_weight, tf.float32)
    l3_weight = tf.constant(l3_weight, tf.float32)
    final_loss = tf.add(tf.add(tf.multiply(l1_weight, k1_loss),
                  tf.multiply(l2_weight, k2_loss)),
                  tf.multiply(l3_weight, classification_loss))

    net['latent_sigmod'] = latent_sigmoid
    net['k1_loss'] = k1_loss
    net['k2_loss'] = k2_loss
    net['classification_loss'] = classification_loss
    net['final_loss'] = final_loss
    net['fc9'] = fc9
    return net, final_loss

def ssdh_net(x, y):
    with tf.device("/gpu:0"):
        with tf.name_scope(name="ssdh"):
            net, tail = _alexnet_head(x, base_decay=FLAGS.weight_decay)
            net, tail = ssdh_body(net,
                            tail,
                            y,
                            num_binary=FLAGS.num_binary,
                            num_class=FLAGS.num_class,
                            base_decay=FLAGS.weight_decay,
                            l1_weight=1,
                            l2_weight=1,
                            l3_weight=1)
    return net, tail


def get_train_op(step):
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
    op1 = tf.train.GradientDescentOptimizer(learning_rate)
    op2 = tf.train.GradientDescentOptimizer(learning_rate * 2).minimize()
    op3 = tf.train.GradientDescentOptimizer(learning_rate * 10)
    op4 = tf.train.GradientDescentOptimizer(learning_rate * 20)

    train_op1 = op1.apply_gradients(zip(g_kernels, kernels))
    train_op2 = op2.apply_gradients(zip(g_biases, biases))
    train_op3 = op3.apply_gradients(zip(g_final_kernel, final_kernel))
    train_op4 = op4.apply_gradients(zip(g_final_biase, final_biase))
    train_op = tf.group(train_op1, train_op2, train_op3, train_op4)
    return train_op

def ssdh_train(net, loss, x, y):
    train_images, train_labels = get_data(FLAGS.train_batch_size, is_train=True)
    test_images, test_labels = get_data(FLAGS.test_batch_size, is_train=False)
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        with tf.device("/gpu:0"):

            sess.run(init)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            import time
            t1 = time.time()
            for step in range(FLAGS.max_iter):
                img, lbs = sess.run([train_images, train_labels])
                train_op = get_train_op(step)
                sess.run(train_op, feed_dict={x:img, y:lbs})

                print step
                if step % FLAGS.print_error_step == 0:
                    t2 = time.time()
                    print "time spend: " + str(t2 - t1)
                    t1 = t2
                    print sess.run(loss, feed_dict={x:img, y:lbs})

                if step % FLAGS.eval_step == 0:
                    print ssdh_evaluation(net, x, y, test_images, test_labels, sess)
        coord.request_stop()
        coord.join(threads)

    pass

def ssdh_evaluation(net, x, y, test_x, test_y, sess):
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    count_predict = 0.0
    count_correct = 0.0
    for i in range(5000/FLAGS.test_batch_size):
        try:
            img, lbs = sess.run([test_x, test_y])
            y_ = sess.run(net['fc9'], feed_dict={x: img, y: lbs})
            with tf.device("/cpu:0"):
                correct_prediction = sess.run(tf.count_nonzero(tf.equal(tf.argmax(lbs, 1), tf.argmax(y_, 1))))
            count_predict += y_.shape[0]
            count_correct += correct_prediction
        except:
            coord.request_stop()
            coord.join(threads)
            break

    coord.request_stop()
    coord.join(threads)
    return count_correct/ count_predict

def get_data(batch_size, is_train=True):
    dir = FLAGS.train_dir if is_train else FLAGS.test_dir
    reader = tf.TFRecordReader()
    filenema_queue = tf.train.string_input_producer([os.path.join(dir, i) for i in os.listdir(dir)])
    _, serialized_example = reader.read(filenema_queue)
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'depth': tf.FixedLenFeature([], tf.int64),
            'label': tf.FixedLenFeature([], tf.string),
            'image_raw': tf.FixedLenFeature([], tf.string)
        })
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image = tf.reshape(image, [height, width, 3])

    label = tf.decode_raw(features['label'], tf.uint8)
    label = tf.reshape(label, [FLAGS.num_class])

    resized_image = tf.image.resize_image_with_crop_or_pad(image=image,
                                                           target_height=FLAGS.dim_size,
                                                           target_width=FLAGS.dim_size)

    min_after_dequeue = 32
    capacity = min_after_dequeue + 3 * batch_size

    if is_train:
        return tf.train.shuffle_batch([resized_image, label],
                                      batch_size=32,
                                      capacity=capacity,
                                      num_threads=2,
                                      min_after_dequeue=min_after_dequeue)
    else:
        return tf.train.batch([resized_image, label],
                                batch_size=batch_size,
                                capacity=50,
                                num_threads=1,
                                allow_smaller_final_batch=True)


if __name__ == "__main__":
    with tf.device("/gpu:0"):
        x = tf.placeholder(dtype=tf.uint8, shape=[None, FLAGS.dim_size, FLAGS.dim_size, 3], name='input')
        y = tf.placeholder(dtype=tf.uint8, shape=[None, FLAGS.num_class], name='label')
        net, loss = ssdh_net(x, y)
    if FLAGS.is_train:
        ssdh_train(net, loss, x, y)
    else:
        ssdh_evaluation(net)
    pass