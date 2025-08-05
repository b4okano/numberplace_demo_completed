
import pandas as pd
from tensorflow import keras
import os
import sys
import cv2
import numpy as np
parent_dir = os.path.abspath("../")
sys.path.append(parent_dir)

def dataset():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # 各label100枚ずつ足りだすためのコード、pandasを用いて行う
    df = pd.DataFrame(columns=["label"])
    df["label"] = y_train.reshape([-1])

    #n=100でsampling
    n=50
    list_1 = df.loc[df.label==1].sample(n)
    list_2 = df.loc[df.label==2].sample(n)
    list_3 = df.loc[df.label==3].sample(n)
    list_4 = df.loc[df.label==4].sample(n)
    list_5 = df.loc[df.label==5].sample(n)
    list_6 = df.loc[df.label==6].sample(n)
    list_7 = df.loc[df.label==7].sample(n)
    list_8 = df.loc[df.label==8].sample(n)
    list_9 = df.loc[df.label==9].sample(n)

    label_list = pd.concat([list_1,list_2,list_3,list_4,list_5,list_6,list_7,list_8,
                        list_9])
    label_list = label_list.sort_index()
    label_idx = label_list.index.values

    train_label = label_list.label.values

    """
    x_trainからlabel用のdataframe.indexを取り出すことでlabelに対応したデータを取り出す。
    """
    x_train = x_train[label_idx]
    y_train= train_label
    x_train = x_train / 255
    x_test = x_test / 255

    empty_dir = "../empty_data"  # 空マス画像が入ったフォルダ
    empty_label = "-"            # 空マスのラベル
    img_size = 28               # 画像のサイズ（28x28）
    #  空マス画像を読み込む
    empty_x = []
    empty_y = []
    for fname in os.listdir(empty_dir):
        if fname.lower().endswith(".jpg"):
            path = os.path.join(empty_dir, fname)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (img_size, img_size))  # MNISTと同じ28x28にリサイズ
            empty_x.append(img)
            empty_y.append(empty_label)

    empty_x = np.array(empty_x)
    empty_y = np.array(empty_y)

    # MNIST と 空マスを結合
    x = np.concatenate([x_train, empty_x])
    y = np.concatenate([y_train, empty_y])
    return x,y