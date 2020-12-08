from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, PReLU, Flatten, Softmax
from tensorflow.keras.models import Model
import tensorflow.compat.v1 as tf
import numpy as np

def PNet():
    
    X = Input(shape = (None, None, 3), name='PNet_Input')
    
    L = Conv2D(10, kernel_size=(3, 3), strides=(1, 1), padding='valid', name='PNet_CONV1')(X)
    L = PReLU(shared_axes=[1, 2], name='PNet_PRELU1')(L)
    L = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='PNet_MAXPOOL1')(L)

    L = Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding='valid', name='PNet_CONV2')(L)
    L = PReLU(shared_axes=[1, 2], name='PNet_PRELU2')(L)

    L = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='valid', name='PNet_CONV3')(L)
    L = PReLU(shared_axes=[1, 2], name='PNet_PRELU3')(L)

    classifier = Conv2D(2, kernel_size=(1, 1), strides=(1, 1), activation='softmax', name = 'FACE_CLASSIFIER')(L)
    regressor = Conv2D(4, kernel_size=(1, 1), strides=(1, 1), name = 'BB_REGRESSION')(L)

    return Model(X, [regressor, classifier], name = 'PNet')

def RNet():

    X = Input(shape = (24,24,3), name='RNet_Input')

    L = Conv2D(28, kernel_size=(3, 3), strides=(1, 1), padding='valid', name='RNet_CONV1')(X)
    L = PReLU(shared_axes=[1, 2], name='RNet_PRELU1')(L)
    L = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='RNet_MAXPOOL1')(L)

    L = Conv2D(48, kernel_size=(3, 3), strides=(1, 1), padding='valid', name='ENet_CONV2')(L)
    L = PReLU(shared_axes=[1, 2], name='RNet_PRELU2')(L)
    L = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid', name='RNet_MAXPOOL2')(L)

    L = Conv2D(64, kernel_size=(2, 2), strides=(1, 1), padding='valid', name='RNet_CONV3')(L)
    L = PReLU(shared_axes=[1, 2], name='RNet_PRELU3')(L)
    L = Flatten(name = 'RNet_FLATTEN')(L)
    L = Dense(128, name = 'RNet_DENSE1')(L)
    L = PReLU(name = 'RNet_PRELU4')(L)

    C = Dense(2, activation='sigmoid', name = 'RNet_DENSE2')(L)
    classifier = Softmax(axis=1, name = 'FACE_CLASSIFIER')(C)
    regressor = Dense(4, name = 'BB_REGRESSION')(L)

    return Model(X, [regressor, classifier], name = 'RNet')


def ONet():
    
    X = Input(shape = (48, 48, 3), name = 'ONet_input')

    L = Conv2D(32, kernel_size= (3,3), strides = (1,1), padding = 'valid', name = 'ONet_CONV1')(X)
    L = PReLU(shared_axes = [1, 2], name = 'ONet_PRELU1')(L)
    L = MaxPooling2D(pool_size = (3,3), strides = (2, 2), padding = 'same', name = 'RNet_MAXPOOL1')(L)
        
    L = Conv2D(64, kernel_size= (3,3), strides = (1,1), padding = 'valid', name = 'ONet_CONV2')(L)
    L = PReLU(shared_axes = [1, 2], name = 'ONet_PRELU2')(L)
    L = MaxPooling2D(pool_size = (3,3), strides = (2, 2), padding = 'valid', name = 'RNet_MAXPOOL2')(L)
        
    L = Conv2D(64, kernel_size= (3,3), strides = (1,1), padding = 'valid', name = 'ONet_CONV3')(L)
    L = PReLU(shared_axes = [1,2], name = 'ONet_PRELU3')(L)
    L = MaxPooling2D(pool_size = (2, 2), strides=(2, 2), padding = 'same', name = 'RNet_MAXPOOL3')(L)
    
    L = Conv2D(128, kernel_size= (2,2), strides = (1, 1), padding = 'valid', name = 'ONet_CONV4')(L)
    L = PReLU(shared_axes = [1, 2], name='ONet_PRELU4')(L)
    
    L = Flatten(name = 'ONet_FLATTEN')(L)
    L = Dense(256, name = 'ONet_DENSE1') (L)
    L = PReLU(name = 'ONet_PRELU5')(L)

    C = Dense(2, activation='sigmoid', name = 'ONet_DENSE2')(L)
    classifier = Softmax(axis = 1, name = 'FACE_CLASSIFIER')(C)
    regressor = Dense(4, name = 'BB_REGRESSION')(L)

    return Model(X, [regressor, classifier], name = 'ONet')


# pnet = PNet()
# pnet.summary()

# rnet = RNet()
# rnet.summary()

# onet = ONet()
# onet.summary()