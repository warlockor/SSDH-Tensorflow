import os
import tensorflow as tf
FLAGS = tf.app.flags.FLAGS

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
    image = tf.decode_raw(features['image_raw'], tf.float32)
    image = tf.reshape(image, [height, width, 3])
    image = tf.random_crop(image, [FLAGS.image_size, FLAGS.image_size, 3])
    label = tf.decode_raw(features['label'], tf.uint8)
    label = tf.reshape(label, [FLAGS.num_class])


    min_after_dequeue = 32
    capacity = min_after_dequeue + 3 * batch_size

    if is_train:
        return tf.train.shuffle_batch([image, label],
                                      batch_size=32,
                                      capacity=capacity,
                                      num_threads=4,
                                      min_after_dequeue=min_after_dequeue)
    else:
        return tf.train.batch([image, label],
                                batch_size=batch_size,
                                capacity=50,
                                num_threads=4,
                                allow_smaller_final_batch=True)