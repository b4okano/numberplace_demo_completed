
import pandas as pd
from tensorflow import keras
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
parent_dir = os.path.abspath("../")
sys.path.append(parent_dir)

def dataset():
        # MNIST 読み込み
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # 0 のラベルのインデックスを取得
    idx_0_train = np.where(y_train == 0)[0]
    idx_0_test = np.where(y_test == 0)[0]

    # 0 の画像を全部真っ白にする（255は白）
    x_train[idx_0_train] = 0
    x_test[idx_0_test] = 0

    # 確認
    plt.imshow(x_train[idx_0_train[0]], cmap="gray")
    plt.title("Modified 0 -> Empty")
    plt.show()