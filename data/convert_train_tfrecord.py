
from absl import app, flags, logging
from absl.flags import FLAGS, argparse_flags
import os
import sys
import numpy as np
import tensorflow as tf
import random
import numpy.random as np_rand
import cv2 as cv
import argparse
import tqdm

FLAGS = flags.FLAGS
flags.DEFINE_string('data_path', './data',
    'path to the training folder')
flags.DEFINE_string('pnet_image_tfrecords', 'pnet_data.tfrecords',
    'file name for pnet records')
flags.DEFINE_string('pnet_bbx_tfrecords', 'pnet_data_bbx.tfrecords',
    'file name for pnet records')

def _bytes_feature(value):
    """ returns a bytes list from a string """
    return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))

def make_image_example(line, label, pixels):
    words = line.split()
    image_file_name = words[0]
    im = cv.imread(image_file_name)
    h, w, ch = im.shape
    if h is not pixels or w is not pixels:
        im = cv.resize(im, (pixels, pixels))
    im = im.astype('uint8')
    label_ = np.array(label, dtype = 'float32')
    label_raw = label_.tostring()
    image_raw = im.tostring()
    feature = {
        'label_raw' : _bytes_feature(label_raw),
        'image_raw' : _bytes_feature(image_raw)
    }
    return tf.train.Example(features = tf.train.Features(feature = feature))

def make_bound_box_example(line, pixels):
    words = line.split()
    image_file_name = words[0]
    im = cv.imread(image_file_name)
    h, w, ch = im.shape
    if h is not pixels or w is not pixels:
        im = cv.resize(im, (pixels, pixels))
    im = im.astype('uint8')
    label = np.array([
        float(words[2]), float(words[3]),
        float(words[4]), float(words[5])], dtype='float32')
    label_raw = label.tostring()
    image_raw = im.tostring()
    feature = {
        'label_raw' : _bytes_feature(label_raw),
        'image_raw' : _bytes_feature(image_raw)
    }
    return tf.train.Example(features = tf.train.Features(feature = feature))

def image_records(files, labels, pixels, output_dir):
    
    samples = []
    if os.path.exists(output_dir):
        logging.info('{:s} already exists. Exit ...'.format(output_dir))
        return

    logging.info('Converting %s sets of images to tfrecords' % len(files))

    length = min([len(i) for i in files])

    for idx, file_ in enumerate(files):
        keep = range(len(file_))
        if len(file_) > length:
            keep = np_rand.choice(len(file_), length, replace = False)
        for i in tqdm.tqdm(keep):
            example = make_image_example(file_[i], labels[idx], pixels)
            samples.append(example)

    logging.info('Writing to records')
    random.shuffle(samples)
    with tf.io.TFRecordWriter(output_dir) as writer:
        for example in tqdm.tqdm(samples):
            writer.write(example.SerializeToString())

def bbx_records(files, pixels, output_dir):
    
    samples = []
    if os.path.exists(output_dir):
        logging.info('{:s} already exists. Exit ...'.format(output_dir))
        return

    logging.info('Converting %s sets of bounding boxes to tfrecords' % len(files))

    length = min([len(i) for i in files])

    for file_ in files:
        keep = range(len(file_))
        if len(file_) > length:
            keep = np_rand.choice(len(file_), length, replace = False)
        for i in tqdm.tqdm(keep):
            example = make_bound_box_example(file_[i], pixels)
            samples.append(example)

    logging.info('Writing to records')
    random.shuffle(samples)
    with tf.io.TFRecordWriter(output_dir) as writer:
        for example in tqdm.tqdm(samples):
            writer.write(example.SerializeToString())


def main(args):

    pixels = args.pixels
    directory = os.path.join(FLAGS.data_path, 'raw_' + str(pixels))
    pos_dir = os.path.join(directory, 'pos/%s.txt' % pixels)
    neg_dir = os.path.join(directory, 'neg/%s.txt' % pixels)
    par_dir = os.path.join(directory, 'part/%s.txt' % pixels)

    # reading lists
    logging.info('Reading data list ...')

    files = []
    for dir in [pos_dir, neg_dir, par_dir]:
        with open(dir, 'r') as f:
            files.append(f.readlines())
    pos, neg, part = files

    # create image records
    image_records([pos, neg], [[0,1],[1,0]], pixels,
        os.path.join(directory, FLAGS.pnet_image_tfrecords))

    # create bounding box records
    bbx_records([pos, part], pixels,
        os.path.join(directory, FLAGS.pnet_bbx_tfrecords))


def parse_arguments(argv):
    parser = argparse_flags.ArgumentParser()
    parser.add_argument('pixels', 
        type=int, 
        help='The side lengths for a generated picture')
    
    return parser.parse_args(argv[1:])

if __name__ == '__main__':        
    try:
        app.run(main, flags_parser=parse_arguments)
    except SystemExit:
        pass