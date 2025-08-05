import numpy as np
from keras import backend as K
from keras.models import load_model
from constant import *
import util
import cv2
import scipy
def predict_board(img, model_path):
    cell_imgs = cells(img)
    model = load_model(model_path)
    y = model.predict(cell_imgs)
    K.clear_session()
    return np.array([np.argmax(c) for c in y])

def cells(img):

    dx = img.shape[0] / 9
    dy = img.shape[1] / 9
    def it():
        for i in range(9):
            for j in range(9):
                sx = int(dx * i)
                sy = int(dy * j)
                cropped = img[sx:(int(sx + dx)), sy:(int(sy + dy))]
                yield util.normalize_img(cropped, IMG_ROWS, IMG_COLS)

    cs = np.array(list(it()))
    cs = cs.reshape(cs.shape[0], IMG_ROWS, IMG_COLS, 1)
    cs = cs.astype(np.float32)
    cs /= 255
    return cs

def cells2(img):
    dx = img.shape[0] / 9  # 行方向の幅
    dy = img.shape[1] / 9  # 列方向の幅
    cs = []

    for i in range(9):
        for j in range(9):
            sx = round(dx * i)
            sy = round(dy * j)
            ex = round(dx * (i + 1))
            ey = round(dy * (j + 1))

            cropped = img[sx:ex, sy:ey]
            cropped = util.normalize_img(cropped, IMG_ROWS, IMG_COLS)
            cs.append(cropped)

    cs = np.array(cs).reshape(-1, IMG_ROWS, IMG_COLS, 1).astype(np.float32) / 255
    return cs

def center_image_on_white(img, white_threshold=1):
    """
    白い画素の重心を計算して、その重心が中心に来るように平行移動する。
    img: 28x28 の numpy 配列（0〜1 のグレースケール画像）
    """
    #assert img.shape == (28, 28)

    # 白画素（＝数字部分）を 1、それ以外を 0 にしたバイナリマスク
    mask = (img >= white_threshold).astype(np.uint8)

    if np.sum(mask) == 0:
        # 白画素がない場合はそのまま返す
        return img.copy()

    # 白画素の重心を計算（重みなし）
    center_of_mass = scipy.ndimage.center_of_mass(mask)

    # 現在の重心と、目標の中心(14,14)との差
    shift_y = 14 - center_of_mass[0]
    shift_x = 14 - center_of_mass[1]

    # 画像を移動（wrapモードではなく、空白部分は0）
    shifted = scipy.ndimage.shift(img.squeeze(axis=2), shift=(shift_y, shift_x), mode='constant', cval=0,order=0)

    return shifted[...,np.newaxis]
