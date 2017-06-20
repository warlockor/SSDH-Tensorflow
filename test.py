import tensorflow as tf

# a = tf.Variable(dtype=tf.float32, initial_value=[[1,2,3],[4,5,6]])
# b = tf.constant([1,1,1], dtype=tf.float32)
# with tf.Session() as sess:
#     sess.run(tf.initialize_all_variables())
#     print sess.run(tf.subtract(a, b))
val = [[[1,2,3], [11, 22, 33]], [[1,2,3], [11, 22, 33]], [[1,2,3], [11, 22, 33]]]
a = tf.constant(val, dtype=tf.float32)

print tf.Session().run(tf.reshape(a, [-1, a.shape._value[1] * a.shape._value[2]]))