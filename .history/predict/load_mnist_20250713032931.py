
from tensorflow import keras
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.layers import Input, Conv2D, Dense, Flatten, MaxPooling2D,Dropout
from keras.models import Sequential

parent_dir = os.path.abspath("../")
sys.path.append(parent_dir)

def dataset(add_noise=False, noise_std=0.1):  # 通常のMNIST読み込み
    # MNIST 読み込み
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    idx_0_train = np.where(y_train == 0)[0]
    idx_0_test = np.where(y_test == 0)[0]

    # 0 の画像を全部真っ黒にする（255は白）
    x_train[idx_0_train] = 0
    x_test[idx_0_test] = 0

    x_train = np.array(x_train, dtype=np.float32) / 255
    x_test = np.array(x_test, dtype=np.float32) / 255

    if add_noise:
        x_train = add_gaussian_noise(x_train, std=noise_std)
        x_test = add_gaussian_noise(x_test, std=noise_std)

    num_classes = 10
    y_train = to_categorical(y_train, num_classes)  # one-hot
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
    Dropout(0.5),
    Dense(num_classes, activation='softmax', name='dense2')
    #Dense(10)
    ])
    return model


def add_gaussian_noise(imgs, mean=0, std=0.1):
    """
    imgs: ndarray (N,28,28), 正規化済み(0〜1)
    mean, std: ノイズの平均・標準偏差
    """
    noise = np.random.normal(mean, std, imgs.shape)
    noisy_imgs = imgs + noise
    noisy_imgs = np.clip(noisy_imgs, 0.0, 1.0)  # 範囲をクリップ
    return noisy_imgs

def add_border_lines(img, line_num=3, border_width=2):
    """
    img: (28,28) 正規化画像 (0~1)
    line_num: 線の本数
    border_width: 外枠に制限
    """
    img = (img * 255).astype(np.uint8).copy()
    h, w = img.shape

    for _ in range(line_num):
        side = np.random.choice(['top', 'bottom', 'left', 'right'])

        if side == 'top':
            y = np.random.randint(0, border_width)
            x1 = np.random.randint(0, w//2)
            x2 = np.random.randint(w//2, w)
            cv2.line(img, (x1,y), (x2,y), 255, 1)

        elif side == 'bottom':
            y = np.random.randint(h-border_width, h)
            x1 = np.random.randint(0, w//2)
            x2 = np.random.randint(w//2, w)
            cv2.line(img, (x1,y), (x2,y), 255, 1)

        elif side == 'left':
            x = np.random.randint(0, border_width)
            y1 = np.random.randint(0, h//2)
            y2 = np.random.randint(h//2, h)
            cv2.line(img, (x,y1), (x,y2), 255, 1)

        else: # right
            x = np.random.randint(w-border_width, w)
            y1 = np.random.randint(0, h//2)
            y2 = np.random.randint(h//2, h)
            cv2.line(img, (x,y1), (x,y2), 255, 1)

    img = img.astype(np.float32) / 255.0
    return img


def dataset_line(add_border_noise=False, line_num=3, border_width=2):
    # MNIST読み込み
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # 0 の画像を全部真っ黒にする
    idx_0_train = np.where(y_train == 0)[0]
    idx_0_test = np.where(y_test == 0)[0]
    x_train[idx_0_train] = 0
    x_test[idx_0_test] = 0

    # 正規化
    x_train = np.array(x_train, dtype=np.float32) / 255
    x_test = np.array(x_test, dtype=np.float32) / 255

    # 🔷 外枠ノイズを加える
    if add_border_noise:
        x_train = np.array([add_border_lines(img, line_num, border_width) for img in x_train])
        x_test = np.array([add_border_lines(img, line_num, border_width) for img in x_test])

    # one-hot
    num_classes = 10
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    return x_train, y_train, x_test, y_test
