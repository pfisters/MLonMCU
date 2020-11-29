from absl import app, flags, logging
from absl.flags import FLAGS

import os
import shutil
import zipfile
import requests

flags.DEFINE_string('data_path', './data/widerface', 
    'path to training dataset')
flags.DEFINE_string('training_file', 'train.zip',
    'name of the training zip file')
flags.DEFINE_string('training_dir', 'train',
    'name of the training directory')
flags.DEFINE_string('validation_file', 'val.zip',
    'name of the validatoin zip file')
flags.DEFINE_string('validation_dir', 'val',
    'name of the validatoin directory')
flags.DEFINE_string('annotation_file', 'labels.zip',
    'name of the annotation zip file')
flags.DEFINE_string('annotation_txt', 'label.txt',
    'name of the annotation text file')
flags.DEFINE_string('annotation_url', 
    'http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/support/bbx_annotation/wider_face_split.zip',
    'url of the annotations')

def download_file_from_google_drive(id, destination):
    
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)


def main(_):

    data_path = FLAGS.data_path
    
    if not os.path.exists(data_path):
        os.mkdir(data_path)
    
    # download wider face training data
    if not os.path.exists(os.path.join(data_path, FLAGS.training_file)) and \
        not os.path.exists(os.path.join(data_path, FLAGS.training_dir)):
        logging.info('downloading training data -- 1.47 GB')
        download_file_from_google_drive(
            "0B6eKvaijfFUDQUUwd21EckhUbWs", 
            os.path.join(data_path, FLAGS.training_file))
    
    # download wider face validation data
    if not os.path.exists(os.path.join(data_path, FLAGS.validation_file)) and \
        not os.path.exists(os.path.join(data_path, FLAGS.validation_dir)):
        logging.info('downloading validation data -- 362.8 MB')
        download_file_from_google_drive(
            "0B6eKvaijfFUDd3dIRmpvSk8tLUk", 
            os.path.join(data_path, FLAGS.validation_file))

    # download annotations
    if not os.path.exists(os.path.join(data_path, FLAGS.annotation_file)):
        logging.info('downloading annotations -- 3.6 MB')
        r = requests.get(FLAGS.annotation_url, allow_redirects = True)
        open(os.path.join(data_path, FLAGS.annotation_file), 'wb').write(r.content)

    # unzip training data and delete zip file
    if not os.path.exists(os.path.join(data_path, FLAGS.training_dir)):
        logging.info('unzipping training data')
        with zipfile.ZipFile(os.path.join(data_path, FLAGS.training_file), 'r') as zip_ref:
            zip_ref.extractall(data_path)
        os.rename(
            os.path.join(data_path, 'WIDER_train'), 
            os.path.join(data_path, FLAGS.training_dir))
        os.remove(os.path.join(data_path, FLAGS.training_file))

    # unzip validation data and delete zip file
    if not os.path.exists(os.path.join(data_path, FLAGS.validation_dir)):
        logging.info('unzipping validation data')
        with zipfile.ZipFile(os.path.join(data_path, FLAGS.validation_file), 'r') as zip_ref:
            zip_ref.extractall(data_path)
        os.rename(
            os.path.join(data_path, 'WIDER_val'), 
            os.path.join(data_path, FLAGS.validation_dir))
        os.remove(os.path.join(data_path, FLAGS.validation_file))

    # unzip labels and delete zip file
    train_path = os.path.join(data_path, FLAGS.training_dir)
    val_path = os.path.join(data_path, FLAGS.validation_dir)
    if not os.path.exists(os.path.join(train_path, FLAGS.annotation_txt)) and \
        not os.path.exists(os.path.join(val_path, FLAGS.annotation_txt)):
        logging.info('unzipping labels')
        with zipfile.ZipFile(os.path.join(data_path, FLAGS.annotation_file), 'r') as zip_ref:
            zip_ref.extractall(data_path)
        labels_dir = os.path.join(data_path, 'wider_face_split')
        os.rename(
            os.path.join(labels_dir, 'wider_face_train_bbx_gt.txt'),
            os.path.join(train_path, FLAGS.annotation_txt))
        os.rename(
            os.path.join(labels_dir, 'wider_face_val_bbx_gt.txt'),
            os.path.join(val_path, FLAGS.annotation_txt))
        shutil.rmtree(labels_dir)
        os.remove(os.path.join(data_path, FLAGS.annotation_file))

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
