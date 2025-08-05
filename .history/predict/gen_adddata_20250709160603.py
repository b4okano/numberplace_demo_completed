import os
import numpy as np
import cv2  # OpenCV
from tqdm import tqdm

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
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # MNIST と同じサイズにリサイズ
        img = cv2.resize(img, (28,28))

        images.append(img)
        labels.append(label)

    # numpy配列にする
    X = np.array(images, dtype=np.uint8)       # shape: (N,28,28)
    y = np.array(labels, dtype=np.uint8)       # shape: (N,)

    return X, y
