from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, PReLU, Flatten, Softmax, ReLU
from tensorflow.keras.models import Model
import tensorflow.compat.v1 as tf
import numpy as np

def PNet():
    
    X = Input(shape = (12, 12, 3), name='PNet_Input')
    
    L = Conv2D(10, kernel_size=(3, 3), strides=(1, 1), padding='valid', name='PNet_CONV1')(X)
    L = ReLU(name='PNet_RELU1')(L)
    L = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='PNet_MAXPOOL1')(L)

    L = Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding='valid', name='PNet_CONV2')(L)
    L = ReLU(name='PNet_RELU2')(L)

    L = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='valid', name='PNet_CONV3')(L)
    L = ReLU(name='PNet_RELU3')(L)

    C = Conv2D(2, kernel_size=(1, 1), strides=(1, 1), name = 'PNet_CONV4')(L)
    classifier = Softmax(axis=1, name = 'FACE_CLASSIFIER')(C)
    regressor = Conv2D(4, kernel_size=(1, 1), strides=(1, 1), name = 'BB_REGRESSION')(L)

    return Model(X, [regressor, classifier], name = 'PNet')


def RNet():

    X = Input(shape = (24,24,3), name='RNet_Input')

    L = Conv2D(28, kernel_size=(3, 3), strides=(1, 1), padding='valid', name='RNet_CONV1')(X)
    L = ReLU(name='RNet_RELU1')(L)
    L = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='RNet_MAXPOOL1')(L)

    L = Conv2D(48, kernel_size=(3, 3), strides=(1, 1), padding='valid', name='ENet_CONV2')(L)
    L = ReLU(name='RNet_RELU2')(L)
    L = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', name='RNet_MAXPOOL2')(L)

    L = Conv2D(64, kernel_size=(2, 2), strides=(1, 1), padding='valid', name='RNet_CONV3')(L)
    L = ReLU(name='RNet_RELU3')(L)
    L = Flatten(name = 'RNet_FLATTEN')(L)
    L = Dense(128, name = 'RNet_DENSE1')(L)
    L = ReLU(name = 'RNet_PRELU4')(L)

    C = Dense(2, name = 'RNet_DENSE2')(L)
    classifier = Softmax(axis=1, name = 'FACE_CLASSIFIER')(C)
    regressor = Dense(4, name = 'BB_REGRESSION')(L)

    return Model(X, [regressor, classifier], name = 'RNet')


def ONet():
    
    X = Input(shape = (48, 48, 3), name = 'ONet_input')

    L = Conv2D(32, kernel_size= (3,3), strides = (1,1), padding = 'valid', name = 'ONet_CONV1')(X)
    L = ReLU(name = 'ONet_RELU1')(L)
    L = MaxPooling2D(pool_size = (3,3), strides = (2, 2), padding = 'same', name = 'RNet_MAXPOOL1')(L)
        
    L = Conv2D(64, kernel_size= (3,3), strides = (1,1), padding = 'valid', name = 'ONet_CONV2')(L)
    L = ReLU(name = 'ONet_RELU2')(L)
    L = MaxPooling2D(pool_size = (3,3), strides = (2, 2), padding = 'valid', name = 'RNet_MAXPOOL2')(L)
        
    L = Conv2D(64, kernel_size= (3,3), strides = (1,1), padding = 'valid', name = 'ONet_CONV3')(L)
    L = ReLU(name = 'ONet_RELU3')(L)
    L = MaxPooling2D(pool_size = (2, 2), strides=(2, 2), padding = 'same', name = 'RNet_MAXPOOL3')(L)
    
    L = Conv2D(128, kernel_size= (2,2), strides = (1, 1), padding = 'valid', name = 'ONet_CONV4')(L)
    L = ReLU(name='ONet_RELU4')(L)
    
    L = Flatten(name = 'ONet_FLATTEN')(L)
    L = Dense(256, name = 'ONet_DENSE1') (L)
    L = ReLU(name = 'ONet_RELU5')(L)

    C = Dense(2, name = 'ONet_DENSE2')(L)
    classifier = Softmax(axis = 1, name = 'FACE_CLASSIFIER')(C)
    regressor = Dense(4, name = 'BB_REGRESSION')(L)

    return Model(X, [regressor, classifier], name = 'ONet')


# pnet = PNet()
# pnet.summary()

# rnet = RNet()
# rnet.summary()

# onet = ONet()
# onet.summary()