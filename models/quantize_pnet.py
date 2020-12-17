from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model
import numpy as np
import json
import cv2
import math

from absl import app, flags, logging
from absl.flags import FLAGS, argparse_flags
from models.MTCNN_models import PNet
from tools.data_handling import get_image_paths, sample_data, load_data, hex_to_c_array
import tensorflow as tf
import os
import tqdm

import argparse

FLAGS = flags.FLAGS
flags.DEFINE_integer('batch_size', 64,
    'batch size for training')
flags.DEFINE_integer('pixels', 12,
    'input size of images', lower_bound=0)
flags.DEFINE_integer('validation_size', 1000,
    'size of the validation set')
flags.DEFINE_list('validation_set_split', [1, 0, 1],
    'split of the validation set: positives, partials, negatives')
flags.DEFINE_string('pnet_weights', './models/pnet.h5',
    'path to the weights of the PNet')
flags.DEFINE_string('pnet_light', './models/pnet.tflite',
    'path to the tf lite model')
flags.DEFINE_bool('save_inputs', False,
    'whether to save the quantized inputs or not')
flags.DEFINE_bool('micro_data', True,
    'input data for the validation of the model on STM32')

def main(args):

    # load model
    model = load_model(FLAGS.pnet_weights)
    model.summary()
    
    # get image paths
    validation_image_paths = get_image_paths(FLAGS.pixels, 'val')

    # get number of training samples
    v_lengths = []
    for path in validation_image_paths:
        (_, anno_path) = path
        anno = open(anno_path, 'r')
        v_lengths.append(len(anno.readlines()))

    ''' 
    calculate the number of samples from the groups:
    positives, partials, negatives 
    '''
    v_split = FLAGS.validation_set_split
    v_numbers = [math.ceil(FLAGS.validation_size / sum(v_split) * s) for s in v_split]

    # make sure there is enought samples
    for l, n in zip(v_lengths, v_numbers):
        if l < n: logging.fatal('There are not enough samples')
    
    # load samples
    v_samples = sample_data(v_numbers, validation_image_paths)

    # load data
    v_data, v_cat, v_bbx = load_data(v_samples, FLAGS.pixels)
    
    if FLAGS.micro_data:
        # convert to csv
        np.savetxt('pnet_data.csv', v_data.reshape(FLAGS.validation_size,-1), delimiter = ',')
        np.savetxt('pnet_cat.csv', v_cat.reshape(FLAGS.validation_size,-1), delimiter=',')
        np.savetxt('pnet_bbx.csv', v_bbx.reshape(FLAGS.validation_size,-1), delimiter=',')

    # converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    quantize = True
    if quantize:
        def representative_dataset():
            for i in tqdm.trange(FLAGS.validation_size):
                yield([v_data[i].reshape(1,FLAGS.pixels,FLAGS.pixels,3)])
            
        # Set the optimization flag.
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # Enforce full-int8 quantization
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8  # or tf.uint8
        converter.inference_output_type = tf.int8  # or tf.uint8
        # Provide a representative dataset to ensure we quantize correctly.
    
    converter.representative_dataset = representative_dataset
    tflitemodel = converter.convert()

    # save converted model
    model_name = 'pnet'
    open(FLAGS.pnet_light, 'wb').write(tflitemodel)
    open(os.path.join('./models', model_name + '.h'), 'w').write(hex_to_c_array(tflitemodel, model_name))

    # analyse full size performance
    score = model.evaluate(
        x=v_data,
        y={
            'FACE_CLASSIFIER' : v_cat,
            'BB_REGRESSION' : v_bbx
            },
        batch_size=FLAGS.batch_size
    )
    full_cat_accuracy = score[5]
    full_bbx_mse = score[4]

    # analyse reduced size performance
    tflite_interpreter = tf.lite.Interpreter(model_path=FLAGS.pnet_light)
    tflite_interpreter.allocate_tensors()
    input_details = tflite_interpreter.get_input_details()
    output_details = tflite_interpreter.get_output_details()

    # empty containers for predictions
    cat_predictions = np.zeros((len(v_data),), dtype=int)
    bbx_predictions = np.zeros((len(v_data), 4), dtype=int)
    intput_scale, input_zero_point = input_details[0]['quantization']
    
    # extract data
    for i in tqdm.trange(len(v_data)):
        val_batch = v_data[i]
        val_batch = val_batch / intput_scale + input_zero_point
        val_batch = np.expand_dims(val_batch, axis=0).astype(input_details[0]['dtype'])

        tflite_interpreter.set_tensor(input_details[0]['index'], val_batch)
        tflite_interpreter.allocate_tensors()
        tflite_interpreter.invoke()

        cat_output = tflite_interpreter.get_tensor(output_details[1]['index'])
        bbx_output = tflite_interpreter.get_tensor(output_details[0]['index'])
        cat_predictions[i] = cat_output.argmax()
        bbx_predictions[i] = bbx_output
    
    # generate test data for micro controller
    if FLAGS.save_inputs:
        filename = 'pnet_test cat|' + '|'.join(map(str,cat_output.flatten())) + '|bbx|' + '|'.join(map(str, bbx_output.flatten()))
        text = np.array2string(val_batch, separator=',', threshold=500).replace('[', '{').replace(']', '}')
        open(filename + '.txt', 'w').write(text)

    # compute bounding box MSE
    bbx_scale, bbx_zero_point = output_details[0]['quantization']
    bbx_predictions_f = (bbx_predictions.astype(float) - bbx_zero_point) * bbx_scale
    
    gt = v_bbx.reshape(bbx_predictions_f.shape)
    quant_bbx_mse = np.square(gt - bbx_predictions_f).mean(axis = 0).mean()

    # compute categorization accuracy
    gt = [i.argmax() for i in v_cat.astype(int)]
    quant_cat_accuracy = sum([1 for p, g in zip(cat_predictions, gt) if p == g]) / len(gt)

    # print result
    logging.info('Classifier Accuracy: ')
    logging.info('Full Model: %s' % full_cat_accuracy)
    logging.info('Quantized Model: %s' % quant_cat_accuracy)
    logging.info('Change in percent: %s' % ((full_cat_accuracy-quant_cat_accuracy) * 100))
    logging.info('Bounding Box MSE:')
    logging.info('Full Model: %s' % full_bbx_mse)
    logging.info('Quantized Model: %s' % quant_bbx_mse)
    logging.info('Change in percent: %s' % ((full_bbx_mse - quant_bbx_mse) * 100))

if __name__ == '__main__':        
    try:
        app.run(main)
    except SystemExit:
        pass
