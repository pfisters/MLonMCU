from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import cv2
import random
import math

from absl import app, flags, logging
from absl.flags import FLAGS, argparse_flags
from models.MTCNN_models import RNet
from tools.data_handling import load_data, sample_data, get_image_paths, read_annotations, preprocess_image, intersection_over_union
from detect_faces import MTCNN
import tensorflow as tf
import os

import argparse
import tqdm

FLAGS = flags.FLAGS
flags.DEFINE_integer('max_imgs', 1000000,
    'maximum number of images in each category')
flags.DEFINE_string('data_path', './data',
    'path to the data')
flags.DEFINE_string('training_path', './data/widerface/train',
    'path to the training folder')
flags.DEFINE_string('validation_path', './data/widerface/val',
    'path to the validation folder')
flags.DEFINE_string('train_annotations', 
    './data/widerface/train/label.txt',
    'path to training annotations')
flags.DEFINE_string('val_annotations', 
    './data/widerface/val/label.txt',
    'path to validation annotations')


def generate_hard_data(pixels, annotations, data_path, im_dir):
    '''
    pixels = 24 / 48 
    annotations = train_annotations
    im_dir = './data/widerface/train/images
    '''
    assert pixels in [24, 48]

    MAX_IMG = FLAGS.max_imgs

    # generate destination folder
    save_dir = os.path.join(FLAGS.data_path, str(pixels))
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # create and open dirs and files
    dirs, files = [], []
    for d in ['pos', 'part', 'neg']:
        _dir = os.path.join(save_dir, d)
        dirs.append(_dir)
        if not os.path.exists(_dir):
            os.mkdir(_dir)
        f = open(os.path.join(save_dir, '%s_%s.txt' % (d, pixels)), 'w')
        files.append(f)

    pos_save_dir, part_save_dir, neg_save_dir = dirs
    f1, f3, f2 = files

    # generate detector
    detector = MTCNN()

    p_idx, n_idx, d_idx, idx = 0, 0, 0, 0

    for anno in tqdm.tqdm(annotations):
        img_path = anno['path']
        img = preprocess_image(os.path.join(im_dir, img_path))
        faces = anno['faces']

        if pixels == 24:
            rectangles = detector.detect_faces_pnet(img)
        if pixels == 48:
            rectangles = detector.detect_faces_rnet(img)

        idx += 1

        for box in rectangles:
            lis = box.astype(np.int32)
            mask = lis < 0
            lis[mask] = 0

            x_left, y_top, x_right, y_bottom, _ = lis
            crop_w = x_right - x_left + 1
            crop_h = y_bottom - y_top + 1
            
            if crop_w < pixels or crop_h < pixels:
                continue
            
            iou = intersection_over_union(box, faces)

            cropped_im = img[y_top: y_bottom + 1, x_left:x_right + 1, :]
            resized_im = cv2.resize(cropped_im, (pixels, pixels), interpolation=cv2.INTER_LINEAR)

            if np.max(iou) < 0.3:
                if n_idx > MAX_IMG:
                    continue
                save_file = os.path.join(neg_save_dir, '%s.jpg' % n_idx)
                f2.write('%s/neg/%s' % (save_dir, n_idx) + ' 0 1\n')
                cv2.imwrite(save_file, resized_im)
                n_idx += 1
            else:
                # find face with the highest iou
                idx = np.argmax(iou)
                assigned_gt = faces[idx]
                x1, x2, y1, y2 = assigned_gt['bb']

                # compute bbx reg label
                offset_x1 = (x1 - x_left) / float(crop_w)
                offset_y1 = (y1 - y_top) / float(crop_h)
                offset_x2 = (x2 - x_right) / float(crop_w)
                offset_y2 = (y2 - y_bottom) / float(crop_h)

                if np.max(iou) >= 0.65:
                    if p_idx > MAX_IMG:
                        continue
                    save_file = os.path.join(pos_save_dir, '%s.jpg' % p_idx)
                    f1.write('%s/pos/%s' % (save_dir, p_idx) + 
                        ' 1 0 %.2f %.2f %.2f %.2f\n' % (
                            offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    p_idx += 1
                elif np.max(iou) >= 0.4:
                    if d_idx > MAX_IMG:
                        continue
                    save_file = os.path.join(part_save_dir, '%s.jpg' % d_idx)
                    f3.write('%s/part/%s' % (save_dir, d_idx) + 
                        ' 0 0 %.2f %.2f %.2f %.2f\n' % (
                            offset_x1, offset_y1, offset_x2, offset_y2))
                    cv2.imwrite(save_file, resized_im)
                    d_idx += 1
    
    for f in files:
        f.close()
    
    logging.info('Generated %s subsets from %s images - pos: %s part: %s neg: %s' %
    (p_idx + d_idx + n_idx, idx, p_idx, d_idx, n_idx))

def main(args):

    pixels = args.pixels
    images = args.images

    # parse annotations
    train_annotations = read_annotations(FLAGS.train_annotations, size=images)
    val_annotations = read_annotations(FLAGS.val_annotations, size =images)

    # data
    generate_hard_data(pixels, 
        train_annotations, 
        FLAGS.data_path, 
        os.path.join(FLAGS.training_path, 'images'))
    

def parse_arguments(argv):
    parser = argparse_flags.ArgumentParser()
    parser.add_argument('pixels', 
        type=int, 
        help='The side lengths for a generated picture')
    parser.add_argument('--images',
        type=int,
        help='number of images to parse',
        default=-1)

    return parser.parse_args(argv[1:])

if __name__ == '__main__':        
    try:
        app.run(main, flags_parser=parse_arguments)
    except SystemExit:
        pass
