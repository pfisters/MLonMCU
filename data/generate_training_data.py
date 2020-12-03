from absl import app, flags, logging
from absl.flags import FLAGS, argparse_flags
import os
import sys
import numpy as np
import numpy.random as np_rand
import cv2 as cv
import argparse
import tqdm

FLAGS = flags.FLAGS
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

def intersection_over_union(box, faces):
    
    bbs = np.array([face['bb'] for face in faces], dtype = np.float32)
    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    area = (bbs[:, 2] - bbs[:, 0] + 1) * (bbs[:, 3] - bbs[:, 1] + 1)
    xx1 = np.maximum(box[0], bbs[:, 0])
    yy1 = np.maximum(box[1], bbs[:, 1])
    xx2 = np.minimum(box[2], bbs[:, 2])
    yy2 = np.minimum(box[3], bbs[:, 3])

    # compute the width and height of the bounding box
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    inter = w * h
    iou = inter / (box_area + area - inter)
    return iou

def read_annotations(path, size = -1):
    logging.info('Loading {}.'.format(path))

    # open file and read each line
    f = open(path, 'r')
    lines = f.readlines()
    # create empty array for training data
    training_data = []
    # iterate over lines
    i = 0
    # create progress bar
    pbar = tqdm.tqdm(total=len(lines))
    while i < len(lines):
        # return if enough samples have been found
        if size > 0 and len(training_data) >= size:
            return training_data
        # initialize new picture
        picture = {}
        # picture must start with file path
        assert lines[i].endswith(".jpg\n"), "read fault " + lines[i]
        picture['path'] = lines[i][:-1] # remove \n character
        # next line contains number of faces
        i += 1
        pbar.update(1)
        if int(lines[i]) == 0:
            i += 2
            pbar.update(2)
            continue
        number_of_faces = max(1, int(lines[i]))
        i += 1
        pbar.update(1)
        faces = []
        for j in range(i, i + number_of_faces):
            face = {}
            features = lines[j]
            features = features.split(' ')
            x, y, w, h = features[:4]
            if int(w) <= 0 or int(h) <= 0: # skip bb with non-positive height or width
                continue
            face['bb'] = [int(x), int(y), int(x) + int(w), int(y) + int(h)]
            face['blur'] = int(features[4])
            face['expression'] = int(features[5])
            face['illumination'] = int(features[6])
            face['occlusion'] = int(features[7])
            face['pose'] = int(features[8])
            face['invalid'] = int(features[9])
            faces.append(face)

        # increase the counter
        i += number_of_faces
        pbar.update(number_of_faces)
        # add picture to training set
        picture['faces'] = faces
        training_data.append(picture)

    pbar.close()
    return training_data

def generate_training_data(pixels, annotations, data_path, im_dir):
    '''
    pixels = 12
    annotations = train_annotations
    im_dir = '.data/widerace/train/images/'
    '''

    directory = os.path.join(data_path, 'raw_' + str(pixels))
    positives = os.path.join(directory, 'pos')
    negatives = os.path.join(directory, 'neg')
    partials = os.path.join(directory, 'part')

    files = []
    # create directories and textfiles
    for dir in [directory, positives, negatives, partials]:
        if not os.path.exists(dir):
            os.mkdir(dir)
        
    for dir in [positives, negatives, partials]:
        if not os.path.exists(os.path.join(dir, str(pixels) + '.txt')):
            f = open(os.path.join(dir, str(pixels) + '.txt'), 'w')
            files.append(f)

    pos_file, neg_file, part_file = files

    idx, p_idx, n_idx, d_idx, face_idx = [0]*5

    for anno in tqdm.tqdm(annotations):

        img_path = anno['path']
        faces = anno['faces']
        img = cv.imread(os.path.join(im_dir, img_path))

        idx += 1
        height, width, channels = img.shape

        # generate negatives
        num_negs = 0
        while num_negs < 50:
            
            size = np_rand.randint(40, min(width, height) / 2)
            x1 = np_rand.randint(0, width - size)
            y1 = np_rand.randint(0, height - size)
            x2 = x1 + size
            y2 = y1 + size

            box = np.array([x1, y1, x2, y2])

            if np.max(intersection_over_union(box, faces)) < 0.3:
                cropped_im = img[y1 : y2, x1 : x2, :]
                resized_im = cv.resize(cropped_im, (pixels, pixels), interpolation = cv.INTER_LINEAR)
                save_path = os.path.join(negatives, '%s.jpg' % n_idx)
                neg_file.write(save_path + ' 0\n')
                cv.imwrite(save_path, resized_im)
                n_idx += 1
                num_negs += 1
        
        # generate positives
        for face in faces:
            x1, y1, x2, y2 = face['bb']
            w = x2 - x1 + 1
            h = y2 - y1 + 1

            if w <= 0 or h <= 0:
                continue

            if max(w,h) < 40 or x1 < 0 or y1 < 0:
                continue

            for _ in range(20):
                size = np_rand.randint(int(min(w,h) * 0.8), np.ceil(1.25 * max(w,h)))
                delta_x = np_rand.randint(-w * 0.2, w * 0.2)
                delta_y = np_rand.randint(-h * 0.2, h * 0.2)

                nx1 = int(max(x1 + w / 2 + delta_x - size / 2, 0))
                ny1 = int(max(y1 + h / 2 + delta_y - size / 2, 0))
                nx2 = nx1 + size
                ny2 = ny1 + size

                if nx2 > width or ny2 > height:
                    continue
                box = np.array([nx1, ny1, nx2, ny2])

                offset_x1 = (x1 - nx1) / float(size)
                offset_y1 = (y1 - ny1) / float(size)
                offset_x2 = (x2 - nx2) / float(size)
                offset_y2 = (y2 - ny2) / float(size)

                cropped_im = img[ny1: ny2, nx1: nx2, :]
                resized_im = cv.resize(cropped_im, (pixels, pixels),interpolation=cv.INTER_LINEAR)
                
                i_o_u = intersection_over_union(box, [face])

                if i_o_u >= 0.65:
                    save_path = os.path.join(positives, '%s.jpg' % p_idx)
                    pos_file.write(save_path + 
                        ' 1 %.2f %.2f %.2f %.2f\n' % \
                        (offset_x1, offset_y1, offset_x2, offset_y2))
                    cv.imwrite(save_path, resized_im)
                    p_idx += 1
                elif i_o_u >= 0.4:
                    save_path = os.path.join(partials, '%s.jpg' % d_idx)
                    part_file.write(save_path + 
                        ' -1 %.2f %.2f %.2f %.2f\n' % \
                        (offset_x1, offset_y1, offset_x2, offset_y2))
                    cv.imwrite(save_path, resized_im)
                    d_idx += 1
            face_idx += 1

    for f in files:
        f.close()

    logging.info('Generated %s subsets from %s images - pos: %s part: %s neg: %s' %
        (p_idx + d_idx + n_idx, idx, p_idx, d_idx, n_idx))

def main(args):

    pixels = args.pixels

    # parse annotations
    train_annotations = read_annotations(FLAGS.train_annotations)
    val_annotations = read_annotations(FLAGS.val_annotations)

    # generate training data
    generate_training_data(pixels, 
        train_annotations, 
        FLAGS.data_path, 
        os.path.join(FLAGS.training_path, 'images'))

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
