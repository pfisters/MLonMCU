from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import cv2
import random
import math

from absl import app, flags, logging
from absl.flags import FLAGS, argparse_flags
from models.MTCNN_models import ONet
from tools.data_handling import get_image_paths, sample_data, load_data
import tensorflow as tf
import os

import argparse
import tqdm

FLAGS = flags.FLAGS
flags.DEFINE_integer('epochs', 20,
    'number of training epochs', lower_bound=0)
flags.DEFINE_integer('batch_size', 16,
    'batch size for training')
flags.DEFINE_float('learning_rate', 1e-3,
    'initial learning rate')
flags.DEFINE_integer('pixels', 48,
    'input size of images', lower_bound=0)
flags.DEFINE_integer('training_size', 50000,
    'size of the trianing set')
flags.DEFINE_integer('validation_size', 20000,
    'size of the validation set')
flags.DEFINE_list('training_set_split', [1, 0, 1,], 
    'split of training set: positives, partials, negatives')
flags.DEFINE_list('validation_set_split', [1, 0, 1],
    'split of the validation set: positives, partials, negatives')


def main(args):

    # get image paths
    training_image_paths = get_image_paths(FLAGS.pixels, 'raw')
    validation_image_paths = get_image_paths(FLAGS.pixels, 'val')

    # get number of training samples
    t_lengths, v_lengths = [], []
    for path in training_image_paths:
        (_, anno_path) = path
        anno = open(anno_path, 'r')
        t_lengths.append(len(anno.readlines()))
    
    for path in validation_image_paths:
        (_, anno_path) = path
        anno = open(anno_path, 'r')
        v_lengths.append(len(anno.readlines()))

    ''' 
    calculate the number of samples from the groups:
    positives, partials, negatives 
    '''
    t_split = FLAGS.training_set_split
    v_split = FLAGS.validation_set_split
    t_numbers = [math.ceil(FLAGS.training_size / sum(t_split) * s) for s in t_split]
    v_numbers = [math.ceil(FLAGS.validation_size / sum(v_split) * s) for s in v_split]

    # make sure there is enought samples
    for l, n in zip(t_lengths, t_numbers):
        if l < n: logging.fatal('There are not enough samples')
    for l, n in zip(v_lengths, v_numbers):
        if l < n: logging.fatal('There are not enough samples')
    
    # load samples
    t_samples = sample_data(t_numbers, training_image_paths)
    v_samples = sample_data(v_numbers, validation_image_paths)

    # load data
    t_data, t_cat, t_bbx = load_data(t_samples, FLAGS.pixels)
    v_data, v_cat, v_bbx = load_data(v_samples, FLAGS.pixels)

    # load model
    model = ONet()

    # define losses
    losses = {
        'FACE_CLASSIFIER' : tf.keras.losses.BinaryCrossentropy(),
        'BB_REGRESSION' : tf.keras.losses.MeanSquaredError()
    }
    loss_weights = {
        'FACE_CLASSIFIER' : .5,
        'BB_REGRESSION' : 1.0
    }

    # compile model
    model.compile(
        loss = losses,
        loss_weights=loss_weights,
        optimizer = tf.keras.optimizers.Adam(
            # learning_rate=FLAGS.learning_rate
            ),
        metrics=['accuracy']
    )

    # train
    H = model.fit(
        x=t_data,
        y={
            'FACE_CLASSIFIER' : t_cat,
            'BB_REGRESSION' : t_bbx
            },
        batch_size=FLAGS.batch_size,
        epochs=FLAGS.epochs
    )

    logging.info('History: ')
    logging.info(H)

    score = model.evaluate(
        x=v_data,
        y={
            'FACE_CLASSIFIER' : v_cat,
            'BB_REGRESSION' : v_bbx
            },
        batch_size=FLAGS.batch_size
    )
    
    # print summary
    model.summary()

    # save model
    model.save(os.path.join('models','onet.h5'))

if __name__ == '__main__':        
    try:
        app.run(main)
    except SystemExit:
        pass
