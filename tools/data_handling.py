from absl import app, flags, logging
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np


def preprocess_image(image_path, size = None):
    if size is not None:
        img = load_img(image_path, target_size=size)
    else:
        img = load_img(image_path)
    # scale image to [0,1]
    img = img_to_array(img) / 255
    # to numpy array
    return np.array(img, dtype='float32')


def resizing_scales(shape):
    if len(shape) is not 3:
        logging.fatal('Invalid argument')
    (h, w, ch) = shape
    
    # initialize scale
    prev_scale = 1.0

    if min(w,h) > 500:
        prev_scale = 500./min(w, h)
    elif max(w,h) < 500:
        prev_scale = 500./max(w, h)

    w = int(w * prev_scale)
    h = int(h * prev_scale)

    # multi scale
    scales = []
    factor = 0.709
    factor_count = 0
    minl = min(h,w)
    while minl >= 12:
        scales.append(prev_scale * pow(factor, factor_count))
        minl *= factor
        factor_count += 1
    
    return scales



    

