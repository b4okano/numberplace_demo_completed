import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Conv2D, Dense, Flatten, MaxPooling2D,Dropout
from keras.models import Sequential

def make_model(input_shape, num_classes): #modelの定義
    model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, name='block1_conv'),
    MaxPooling2D((2, 2), name='block1_pool'),

    Conv2D(64, (3, 3), activation='relu', name='block2_conv'),
    MaxPooling2D((2, 2), name='block2_pool'),

    Conv2D(64, (3, 3), activation='relu', name='block3_conv'),
    MaxPooling2D((2, 2), name='block3_pool'),

    Flatten(name='flatten'),
    Dense(128, activation='relu', name='dense1'),
    #Dropout(0.5),
    Dense(num_classes, activation='softmax', name='dense2')
    #Dense(10)
    ])
    return model