import tensorflow as tf

# a = tf.Variable(dtype=tf.float32, initial_value=[[1,2,3],[4,5,6]])
# b = tf.constant([1,1,1], dtype=tf.float32)
# with tf.Session() as sess:
#     sess.run(tf.initialize_all_variables())
#     print sess.run(tf.subtract(a, b))

a = tf.constant([1,2,3], dtype=tf.float32)
print tf.Session().run(tf.square(a))