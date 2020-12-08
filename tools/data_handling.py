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

