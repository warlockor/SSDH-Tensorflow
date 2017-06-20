import tensorflow as tf
import time
tf.app.flags.DEFINE_integer("dim_size", 256, "image size into netword")

tf.app.flags.DEFINE_integer("image_size", 227, "image size into netword")
tf.app.flags.DEFINE_string("mean_file", "./data/ilsvrc_2012_mean.npy", "mean file")
tf.app.flags.DEFINE_integer("num_class", 3, "classification number")
tf.app.flags.DEFINE_integer("num_binary", 800, "number of binary code")
tf.app.flags.DEFINE_float("weight_decay", 0.0005, "l2 weight regularization decay")
tf.app.flags.DEFINE_string("train_dir", "./data/train", "train data dir")
tf.app.flags.DEFINE_string("test_dir", "./data/test", "train data dir")

tf.app.flags.DEFINE_bool("is_train", True, "train or test")
tf.app.flags.DEFINE_integer("train_batch_size", 32, "batch size in train")
tf.app.flags.DEFINE_integer("test_batch_size", 50, "batch size in test")

tf.app.flags.DEFINE_float("learning_rate", 0.001, "learning rate")
tf.app.flags.DEFINE_integer("max_iter", 50000, "max iterator number")
tf.app.flags.DEFINE_integer("print_error_step", 100, 'print_error_step')
tf.app.flags.DEFINE_integer("eval_step", 1000, 'print_error_step')

tf.app.flags.DEFINE_integer("lr_decay_step", 10000, 'learning decay step')
tf.app.flags.DEFINE_float("lr_decay", 0.1, 'learning decay step')

FLAGS = tf.app.flags.FLAGS

from train_net import ssdh_net
from fetch_data import get_data


def ssdh_eval(net, x, y, test_image, test_y):
    count_predict = 0.0
    count_correct = 0.0
    for _ in range(5000 / FLAGS.test_batch_size):
        img, lbs = sess.run([test_image, test_y])
        y_ = sess.run(net['fc9'], feed_dict={x: img, y: lbs})
        with tf.device("/cpu:0"):
            correct_prediction = sess.run(tf.reduce_sum(tf.cast(tf.equal(tf.argmax(lbs, 1), tf.argmax(y_, 1)), tf.float32)))
        count_predict += y_.shape[0]
        count_correct += correct_prediction


    return count_correct / count_predict


if __name__ == "__main__":
    train_images, train_labels = get_data(FLAGS.train_batch_size, is_train=True)
    test_images, test_labels = get_data(FLAGS.test_batch_size, is_train=False)

    global_step = tf.Variable(0, trainable=False, dtype=tf.int16)
    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate,
                                               global_step,
                                               FLAGS.lr_decay_step,
                                               FLAGS.lr_decay,
                                               staircase=True)


    with tf.device("/gpu:0"):
        x = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.image_size, FLAGS.image_size, 3], name='input')
        y = tf.placeholder(dtype=tf.uint8, shape=[None, FLAGS.num_class], name='label')
        net, loss = ssdh_net(x, y)


        init = tf.initialize_all_variables()

        with tf.Session() as sess:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

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

            t1 = time.time()

            op1 = tf.train.GradientDescentOptimizer(tf.multiply(learning_rate, 1))
            op2 = tf.train.GradientDescentOptimizer(tf.multiply(learning_rate, 2))
            op3 = tf.train.GradientDescentOptimizer(tf.multiply(learning_rate, 10))
            op4 = tf.train.GradientDescentOptimizer(tf.multiply(learning_rate, 20))

            train_op1 = op1.apply_gradients(zip(g_kernels, kernels))
            train_op2 = op2.apply_gradients(zip(g_biases, biases))
            train_op3 = op3.apply_gradients(zip(g_final_kernel, final_kernel))
            train_op4 = op4.apply_gradients(zip(g_final_biase, final_biase))
            train_op = tf.group(train_op1, train_op2, train_op3, train_op4)

            sess.run(init)
            for step in range(FLAGS.max_iter):

                sess.run(global_step.assign(step))
                img, lbs = sess.run([train_images, train_labels])
                _, total_loss, k1_loss, k2_loss, cls_loss = sess.run([train_op,
                                                                      loss,
                                                                      net['k1_loss'],
                                                                      net['k2_loss'],
                                                                      net['classification_loss']],
                                                                     feed_dict={x: img, y: lbs})

                if step % FLAGS.print_error_step == 0:
                    y_ = sess.run(net['fc9'], feed_dict={x: img, y: lbs})
                    with tf.device("/cpu:0"):
                        print sess.run(tf.reduce_mean(tf.cast(tf.equal(tf.argmax(lbs, 1), tf.argmax(y_, 1)), tf.float32)))

                    print "step: {num_step}, loss: {loss}, k1: {k1}, k2: {k2}, cls_loss: {cls_loss}"\
                        .format(num_step=step,
                                loss=total_loss,
                                k1=k1_loss,
                                k2=k2_loss,
                                cls_loss=cls_loss)

                if step % FLAGS.eval_step == 0:
                    print ssdh_eval(net, x, y, test_images, test_labels)


            coord.request_stop()
            coord.join(threads)