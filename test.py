import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
with tf.device("/gpu:1"):
    a = tf.constant([1, 2, 3, 4, 5, 6, 7, 8, 9], shape=[3,3])
    b = tf.constant([1, 2, 3, 4, 5, 6, 7, 8, 9], shape=[3,3])
    for i in xrange(100000):
        c = tf.multiply(a, b)

    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        print sess.run(c)