import os
import numpy as np
import cv2  # OpenCV
from tqdm import tqdm
from tensorflow import keras
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def load_folder_as_mnist(folder):
    images = []
    labels = []

    # フォルダ内のファイルをループ
    for fname in tqdm(os.listdir(folder)):
        if not fname.endswith('.png'):
            continue
        
        # ファイル名からラベルを抽出
        # 例: img_0001_5.png → 5
        label_str = fname.split('_')[-1].replace('.png', '')
        label = int(label_str)

        # 画像を読み込み & グレースケールに変換
        img_path = os.path.join(folder, fname)
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # ノイズ除去（ガウシアンブラー）
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            # 適応的二値化（背景が不均一な場合に強い）
        img = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11, 4
            )
        # _, img = cv2.threshold(
        # blurred, 0, 255,
        # cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        # )

        
        # MNIST と同じサイズにリサイズ
        img = cv2.resize(img, (28,28))

        images.append(img)
        labels.append(label)

    # numpy配列にする
    x = np.array(images, dtype=np.uint8)       # shape: (N,28,28)
    y = np.array(labels, dtype=np.uint8)       # shape: (N,)
    num_classes = 10
    x = x[..., np.newaxis]  # チャネル次元を追加
    y = to_categorical(y, num_classes) #one-hot
    ex_train, val_ex_train, ey_train, val_ey_train = train_test_split(x, y, train_size=0.8, random_state=0) #train

    return ex_train, val_ex_train, ey_train, val_ey_train
