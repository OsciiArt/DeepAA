"""
Build deep neural network model.
"""

from keras.models import  Model
from keras.layers import Dense, Activation, Reshape, Flatten, Dropout, Input, GaussianNoise, BatchNormalization
from keras.layers import Convolution2D, MaxPooling2D
from keras.initializations import normal
from keras.regularizers import l2


def DeepAA(num_label=615, drop_out=0.5, weight_decay=0.001, input_shape = [64, 64]):
    """
    Build Deep Neural Network.
    :param num_label: int, number of classes, equal to candidates of characters
    :param drop_out:  float
    :param weight_decay: float
    :return: 
    """
    reg = l2(weight_decay)
    imageInput = Input(shape=input_shape)
    x = Reshape([input_shape[0], input_shape[1], 1])(imageInput)
    x = GaussianNoise(0.1)(x)
    x = Convolution2D(16, 3, 3, border_mode='same', W_regularizer=reg, b_regularizer=reg, init=normal)(x)
    x = BatchNormalization(axis=-3)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), border_mode='same')(x)
    x = Dropout(drop_out)(x)
    x = Convolution2D(32, 3, 3, border_mode='same', W_regularizer=reg, b_regularizer=reg, init=normal)(x)
    x = BatchNormalization(axis=-3)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), border_mode='same')(x)
    x = Dropout(drop_out)(x)
    x = Convolution2D(64, 3, 3, border_mode='same', W_regularizer=reg, b_regularizer=reg, init=normal)(x)
    x = BatchNormalization(axis=-3)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), border_mode='same')(x)
    x = Dropout(drop_out)(x)
    x = Convolution2D(128, 3, 3, border_mode='same', W_regularizer=reg, b_regularizer=reg, init=normal)(x)
    x = BatchNormalization(axis=-3)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), border_mode='same')(x)
    x = Flatten()(x)
    x = Dropout(drop_out)(x)
    y = Dense(num_label, activation='softmax')(x)

    model = Model(input=imageInput, output=y)
    
    return model