import numpy as np
import sys
from tqdm import tqdm


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
    pbar = tqdm(total=len(lines))
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
        number_of_faces = max(1, int(lines[i]))
        i += 1
        pbar.update(1)
        faces = []
        for j in range(i, i + number_of_faces):
            face = {}
            features = lines[j]
            features = features.split(' ')
            x, y, w, h = features[:4]
            face['bb'] = [int(x),int(y), int(x) + int(w), int(y) + int(h)]
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