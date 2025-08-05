
from tensorflow import keras
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
parent_dir = os.path.abspath("../")
sys.path.append(parent_dir)

def dataset():
        # MNIST 読み込み
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    idx_0_train = np.where(y_train == 0)[0]
    idx_0_test = np.where(y_test == 0)[0]

            # 0 の画像を全部真っ白にする（255は白）
    x_train[idx_0_train] = 255
    x_test[idx_0_test] = 255

    #plt.imshow(x_train[idx_0_train[0]], cmap="gray",vmin=0,vmax=255)
    #plt.title("Modified 0 -> Empty")
    #plt.show()

    for i in range(1,10):
            # 1から9のラベルのインデックスを取得
        idx_train = np.where(y_train == i)[0]
        idx_test = np.where(y_test == i)[0]

            # 0 の画像を反転する（255は白）
        x_train[idx_train] = 255 - x_train[idx_train]
        x_test[idx_test] = 255 - x_test[idx_test]

        # 確認
        #plt.imshow(x_train[idx_train[0]], cmap="gray")
        #plt.title("Modified 0 -> Empty")
        #plt.show()
    x_train = np.array(x_train, dtype=np.float32) / 255
    x_test = np.array(x_test, dtype=np.float32) / 255

    num_classes = 10
    y_train = to_categorical(y_train, num_classes) #one-hot
    y_test = to_categorical(y_test, num_classes)
    return x_train, y_train, x_test, y_test