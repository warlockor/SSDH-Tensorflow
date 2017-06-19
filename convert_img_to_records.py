from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import random
import tensorflow as tf
import numpy as np

tf.app.flags.DEFINE_string("data_dir", "/home/mozat-1/ssdh_data/subcategory_v4/17_Dresses", "data folder store the origin images")
tf.app.flags.DEFINE_integer("store_batch_size", 6000, "tfrecord batch size")
tf.app.flags.DEFINE_string("directory", "/home/mozat-1/PycharmProjects/SSDH-Tensorflow/data", "the directory store tfrecord")
tf.app.flags.DEFINE_integer("validation_size", 5000, "validataion data set")
tf.app.flags.DEFINE_integer("image_size", 256, "image size")
tf.app.flags.DEFINE_string("mean_file", "./data/ilsvrc_2012_mean.npy", "mean file")

FLAGS = tf.app.flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)

mean_file = np.load(FLAGS.mean_file)


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def convert(images, labels, name):
    if images.shape[0] != len(labels):
        raise ValueError('Images size %d does not match label size %d.' %
                         (images.shape[0], labels.shape[0]))
    num_examples = labels.shape[0]
    rows = images.shape[1]
    cols = images.shape[2]
    depth = images.shape[3]

    filename = os.path.join(FLAGS.directory, name + '.tfrecords')
    writer = tf.python_io.TFRecordWriter(filename)
    for index in range(num_examples):
        img_trans = np.transpose(images[index], [2, 0, 1])[::-1,:,:] - mean_file
        img = np.transpose(img_trans, [1, 2, 0])
        image_raw = img.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(rows),
            'width': _int64_feature(cols),
            'depth': _int64_feature(depth),
            'label': _bytes_feature(np.asarray([0] * labels[index] + [1] + [0]*(3-1 - labels[index]), dtype=np.uint8).tostring()),
            'image_raw': _bytes_feature(image_raw)}))
        writer.write(example.SerializeToString())
    writer.close()


def img_filename_and_label():
    data_dir = FLAGS.data_dir

    filename_lables = []

    labels_space = os.listdir(data_dir)

    for label in os.listdir(data_dir):
        img_dir = os.path.join(data_dir, label)
        for filename in os.listdir(img_dir):
            img_path = os.path.join(img_dir, filename)
            filename_lables.append((img_path, labels_space.index(label)))

    random.shuffle(filename_lables)
    filenames = map(lambda x: x[0], filename_lables)
    labels = map(lambda x: x[1], filename_lables)
    return filenames, labels

def data_batch(filenames, labels):
    store_batch_size = FLAGS.store_batch_size
    for i in range(0,len(filenames), store_batch_size):
        filenames_batch = filenames[i:i+store_batch_size]
        images = np.vstack([[cv2.resize(cv2.imread(img),
                                        (FLAGS.image_size, FLAGS.image_size))]
                            for img in filenames_batch])
        labels_batch = np.asarray(labels[i:i+store_batch_size])
        yield (images, labels_batch)
    pass


def main(unused_argv):
    filenames, labels = img_filename_and_label()
    tr_filenames = filenames[FLAGS.validation_size:]
    tr_labels = labels[FLAGS.validation_size:]
    ts_filenames = filenames[:FLAGS.validation_size]
    ts_labels = labels[:FLAGS.validation_size]

    tf.logging.info("Start convert training dataset")
    for index, (X, Y) in enumerate(data_batch(tr_filenames, tr_labels)):
        tf.logging.info("train{index}".format(index=str(index).zfill(5)))
        convert(X, Y, "train{index}".format(index=str(index).zfill(5)))

    tf.logging.info("Start convert testing dataset")
    for index, (X, Y) in enumerate(data_batch(ts_filenames, ts_labels)):
        tf.logging.info("Test{index}".format(index=str(index).zfill(5)))
        convert(X, Y, "Test{index}".format(index=str(index).zfill(5)))
    pass

if __name__ == "__main__":
    tf.app.run(main=main)
    pass