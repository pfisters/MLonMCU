from absl import app, flags, logging
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os
import random
import tqdm


FLAGS = flags.FLAGS

def preprocess_image(image_path, size = None):
    if size is not None:
        img = load_img(image_path, target_size=size)
    else:
        img = load_img(image_path)
    # scale image to [0,1]
    img = img_to_array(img) / 255
    # to numpy array
    return np.array(img, dtype='float32')


def get_image_paths(pixels, identifier):
    logging.info('Get image paths for %s and %s pixel squares' % (identifier, pixels))
    image_path = os.path.join('data', '%s_%s' % (identifier, pixels))
    image_paths = []
    for i in ['pos', 'part', 'neg']:
        im_path = os.path.join(image_path, i)
        anno_path = os.path.join(im_path, '%s.txt' % pixels)
        image_paths.append((im_path, anno_path))

    return image_paths

def sample_data(numbers, paths):
    logging.info('Sample data from %s' % paths)
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
        # scale image to [-1,1]
        img = img_to_array(img) / 255
        # append to images
        data_.append(img)
        # get category
        cat = sample.split(' ')[1:3]
        cat = [int(x) for x in cat]
        cls_.append(cat)
        # get bounding box
        if cat[1] is not 1:
            bbx = sample.split(' ')[3:]
            bbx = [float(x) for x in bbx]
            bbx_.append((bbx[0], bbx[1], bbx[2], bbx[3]))
        else:
            bbx_.append((0.0,0.0,0.0,0.0))
    
    # convert to numpy array
    data = np.array(data_, dtype='float32')
    cat = np.array(cls_, dtype='float32')
    bbx = np.array(bbx_, dtype='float32')

    # reshape to have the correct output shape for the pnet
    if pixels == 12:
        bbx = bbx.reshape(bbx.shape[0], 1, 1, -1)
        cat = cat.reshape(cat.shape[0], 1, 1, -1)

    # return
    return data, cat, bbx

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
            bbx = features[:4]
            # convert to integers
            bbx = [int(i) for i in bbx]
            if bbx[2] <= 0 or bbx[3] <= 0: # skip bb with non-positive height or width
                continue
            face['bb'] = [bbx[0], bbx[1], bbx[0] + bbx[2], bbx[1] + bbx[3]]
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

def intersection_over_union(box, faces):
    bbs = np.array([face['bb'] for face in faces], dtype = np.float32)
    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    if bbs.size == 0:
        return None
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
