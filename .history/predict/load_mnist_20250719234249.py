
from tensorflow import keras
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras.layers import Input, Conv2D, Dense, Flatten, MaxPooling2D,Dropout
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from skimage.util import random_noise

parent_dir = os.path.abspath("../")
sys.path.append(parent_dir)

# def dataset(add_noise=False, noise_std=0.1):  # 通常のMNIST読み込み
#     # MNIST 読み込み
#     (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

#     idx_0_train = np.where(y_train == 0)[0]
#     idx_0_test = np.where(y_test == 0)[0]

#     # 0 の画像を全部真っ黒にする（255は白）
#     x_train[idx_0_train] = 0
#     x_test[idx_0_test] = 0

#     x_train = np.array(x_train, dtype=np.float32) / 255
#     x_test = np.array(x_test, dtype=np.float32) / 255

#     if add_noise:
#         x_train = add_gaussian_noise(x_train, std=noise_std)
#         x_test = add_gaussian_noise(x_test, std=noise_std)

#     num_classes = 10
#     y_train = to_categorical(y_train, num_classes)  # one-hot
#     y_test = to_categorical(y_test, num_classes)
#     return x_train, y_train, x_test, y_test

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


# def add_gaussian_noise(imgs, mean=0, std=0.1):
#     """
#     imgs: ndarray (N,28,28), 正規化済み(0〜1)
#     mean, std: ノイズの平均・標準偏差
#     """
#     noise = np.random.normal(mean, std, imgs.shape)
#     noisy_imgs = imgs + noise
#     noisy_imgs = np.clip(noisy_imgs, 0.0, 1.0)  # 範囲をクリップ
#     return noisy_imgs

def add_border_lines(img, line_num=5, border_width=3):
    """
    img: (28,28) 正規化画像 (0~1)
    line_num: 線の本数
    border_width: 外枠に制限
    """
    img = (img * 255).astype(np.uint8).copy()
    h, w = img.shape
    line_num = np.random.randint(0,line_num)


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

def add_salt_and_pepper(img, amount=3):
    """
    1枚の画像にごま塩ノイズを加える
    img: 2次元numpy配列（0〜1に正規化済みのMNIST画像）
    amount: ノイズの割合（例: 0.01 → 全ピクセルの1%にノイズ）
    """
    amount = np.random.randint(0, amount) * 0.01
    noisy = random_noise(img, mode='s&p', amount=amount)
    # skimageのrandom_noiseはfloat64で返すので、元と同じdtypeにして返す
    return noisy.astype(np.float32) / 255.0


def dataset_line(add_border_noise=False, line_num=3, border_width=3,sp_amount=0.0):
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
        x_train, val_x_train, y_train, val_y_train = train_test_split(x_train, y_train, train_size=0.5, random_state=0) #train
        x_test, val_x_test, y_test, val_y_test = train_test_split(x_test, y_test, train_size=0.5, random_state=0) #train

        x_train = np.array([add_border_lines(img, line_num, border_width) for img in x_train])
        x_test = np.array([add_border_lines(img, line_num, border_width) for img in x_test])

        x_train = np.array([add_salt_and_pepper(img, sp_amount) for img in x_train])
        x_test = np.array([add_salt_and_pepper(img, sp_amount) for img in x_test])

        # 結合
        x_train = np.concatenate([x_train, val_x_train], axis=0)
        y_train = np.concatenate([y_train, val_y_train], axis=0)
        x_test = np.concatenate([x_test, val_x_test], axis=0)
        y_test = np.concatenate([y_test, val_y_test], axis=0)

        #同じ順序でシャッフルする
        indices = np.arange(len(x_train))
        np.random.shuffle(indices)

        x_train = x_train[indices]
        y_train = y_train[indices]

        indices = np.arange(len(x_test))
        np.random.shuffle(indices)
        x_test = x_test[indices]
        y_test = y_test[indices]


    # one-hot
    num_classes = 10
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    return x_train, y_train, x_test, y_test
