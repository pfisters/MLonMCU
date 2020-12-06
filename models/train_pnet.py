from tensorflow.keras.preprocessing.image import img_to_array, load_img
import tensorflow.keras.backend as K
import numpy as np
import cv2
import random
import math

from absl import app, flags, logging
from absl.flags import FLAGS, argparse_flags
from models.MTCNN_models import PNet
import tensorflow as tf
import os

import argparse
import tqdm

FLAGS = flags.FLAGS
flags.DEFINE_integer('epochs', 50,
    'number of training epochs', lower_bound=0)
flags.DEFINE_integer('batch_size', 64,
    'batch size for training')
flags.DEFINE_float('learning_rate', 1e-3,
    'initial learning rate')
flags.DEFINE_integer('pixels', 12,
    'input size of images', lower_bound=0)
flags.DEFINE_integer('training_size', 60000,
    'size of the trianing set')
flags.DEFINE_integer('validation_size', 20000,
    'size of the validation set')
flags.DEFINE_list('training_set_split', [1, 0, 1,], 
    'split of training set: positives, partials, negatives')
flags.DEFINE_list('validation_set_split', [1, 0, 1],
    'split of the validation set: positives, partials, negatives')


def get_image_paths(pixels, identifier):
    logging.info('Get image paths')
    image_path = os.path.join('data', '%s_%s' % (identifier, FLAGS.pixels))
    image_paths = []
    for i in ['pos', 'part', 'neg']:
        im_path = os.path.join(image_path, i)
        anno_path = os.path.join(im_path, '%s.txt' % FLAGS.pixels)
        image_paths.append((im_path, anno_path))

    return image_paths

def sample_data(numbers, paths):
    logging.info('Sample data')
    samples = []

    for n, p in zip(numbers, paths):
        (_, anno_path) = p
        anno = open(anno_path, 'r')
        lines = anno.readlines()
        sample = random.choices(lines, k=n)
        samples += sample

    random.seed(42)
    random.shuffle(samples)

    return samples

def load_data(samples, pixels):
    logging.info('Load data into memory')
    data_ = []
    cls_ = []
    bbx_ = []

    for sample in tqdm.tqdm(samples):
        # remove new line character
        sample = sample[:-1]
        image_path = sample.split(' ')[0]
        # load and scale image
        img = load_img(image_path, target_size=(pixels, pixels))
        # scale image to [0,1]
        img = img_to_array(img) / 255.0
        # append to images
        data_.append(img)
        # get category
        image_cat = int(sample.split(' ')[1])
        cls_.append(image_cat)
        # get bounding box
        if image_cat is not 0:
            [x1, y1, x2, y2] = sample.split(' ')[2:]
            bbx_.append((float(x1), float(y1), float(x2), float(y2)))
        else:
            bbx_.append((0.0,0.0,0.0,0.0))
    
    # convert to numpy array
    data = np.array(data_, dtype='float32')
    cat = np.array(cls_, dtype='float32')
    bbx = np.array(bbx_, dtype='float32')

    # reshape to have the correct output shape for the pnet
    bbx = bbx.reshape(bbx.shape[0], 1, 1, -1)
    cat = cat.reshape(cat.shape[0], 1, 1, -1)

    # return
    return data, cat, bbx

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
    model = PNet()

    # define losses
    losses = {
        'FACE_CLASSIFIER' : tf.keras.losses.BinaryCrossentropy(),
        'BB_REGRESSION' : tf.keras.losses.MeanSquaredError()
    }
    loss_weights = {
        'FACE_CLASSIFIER' : 1.0,
        'BB_REGRESSION' : 1.0
    }

    # compile model
    model.compile(
        loss = losses,
        loss_weights=loss_weights,
        optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate),
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
    model.save(os.path.join('models','pnet.h5'))
    # model.save_weights(os.path.join('models','pnet.h5'))

if __name__ == '__main__':        
    try:
        app.run(main)
    except SystemExit:
        pass
