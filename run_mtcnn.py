from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import random
import math

from absl import app, flags, logging
from absl.flags import FLAGS, argparse_flags
from models.MTCNN_models import PNet, RNet, ONet
from tools.data_handling import preprocess_image, resizing_scales
import tensorflow as tf
import os

import argparse

FLAGS = flags.FLAGS
flags.DEFINE_string('pnet_weights', './models/pnet.h5',
    'path to the weights of the PNet')
flags.DEFINE_string('rnet_weights', './models/rnet.h5',
    'path to the weights of the RNet')
flags.DEFINE_string('onet_weights', './models/onet.h5',
    'path to the weights of the ONet')


def main(args):

    # check input
    image_path = args.image
    if not os.path.exists(image_path):
        logging.fatal('Image does not exists')
    
    # load models
    pnet = load_model(FLAGS.pnet_weights)
    rnet = load_model(FLAGS.rnet_weights)
    onet = load_model(FLAGS.onet_weights)

    # load image
    image = preprocess_image(image_path)
    
    # compute resizing scales
    (orig_h, orig_w, ch) = image.shape
    scales = resizing_scales(image.shape)
    
    # run through pnet
    pnet_out = []
    for scale in scales:
        new_h = int(orig_h * scale)
        new_w = int(orig_w * scale)
        scaled_img = preprocess_image(image_path, (new_h, new_w))
        scaled_img = scaled_img.reshape(1,*scaled_img.shape)
        out = pnet.predict(scaled_img)
        pnet_out.append(out)

    print(pnet_out[0])

def parse_arguments(argv):
    parser = argparse_flags.ArgumentParser()
    parser.add_argument('image', help='path to the image')

    return parser.parse_args(argv[1:])

if __name__ == '__main__':        
    try:
        app.run(main, flags_parser=parse_arguments)
    except SystemExit:
        pass


