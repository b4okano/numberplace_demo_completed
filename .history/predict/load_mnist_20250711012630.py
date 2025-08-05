
from tensorflow import keras
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.layers import Input, Conv2D, Dense, Flatten, MaxPooling2D,Dropout
from keras.models import Sequential

parent_dir = os.path.abspath("../")
sys.path.append(parent_dir)

def dataset():
        # MNIST 読み込み
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    idx_0_train = np.where(y_train == 0)[0]
    idx_0_test = np.where(y_test == 0)[0]

            # 0 の画像を全部真っ黒にする（255は白）
    x_train[idx_0_train] = 0
    x_test[idx_0_test] = 0

    #plt.imshow(x_train[idx_0_train[0]], cmap="gray",vmin=0,vmax=255)
    #plt.title("Modified 0 -> Empty")
    #plt.show()


    x_train = np.array(x_train, dtype=np.float32) / 255
    x_test = np.array(x_test, dtype=np.float32) / 255

    num_classes = 10
    y_train = to_categorical(y_train, num_classes) #one-hot
    y_test = to_categorical(y_test, num_classes)
    return x_train, y_train, x_test, y_test

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
    Dropout(0.3),
    Dense(num_classes, activation='softmax', name='dense2')
    #Dense(10)
    ])
    return model