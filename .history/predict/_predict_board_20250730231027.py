import numpy as np
from keras import backend as K
from keras.models import load_model
import util
import cv2
import scipy
import matplotlib.pyplot as plt


def predict_board(img, model_path):
    cell_imgs = cells(img)
    model = load_model(model_path)
    y = model.predict(cell_imgs)
    K.clear_session()
    return np.array([np.argmax(c) for c in y])

def cells(img):
    IMG_ROWS = 100
    IMG_COLS = 100

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
    IMG_ROWS = 100
    IMG_COLS = 100
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



def showfig(label):#モデルによる推測結果の表示
    plt.figure(figsize=(8,8))   
    plt.title("Prediction Results",loc='left')
    tb = plt.table(cellText=label,
                    loc='left',
                    cellLoc='center')
    tb.scale(0.5, 1.5)
    plt.axis('off')

    cell_width = tb[0, 0].get_width()
    cell_height = tb[0, 0].get_height()

    # 行ラベルを枠なしで追加
    for i in range(9):
        cell = tb.add_cell(i, -1, width=cell_width*1.3, height=cell_height,
                        text=str(i+1), loc='center')
        cell.set_edgecolor('white')  # 枠線を消す

    # 列ラベルを枠なしで追加
    for j in range(9):
        cell = tb.add_cell(-1, j, width=cell_width, height=cell_height*1.3,
                        text=str(j+1), loc='center')
        cell.set_edgecolor('white')  # 枠線を消す
    
    plt.show()

def draw_labels_on_image(img, labels, size=9): #labels: size * sizeの配列
    h, w, c = img.shape
    h = h // size
    w = w // size
    h_p = int(h*0.3)
    w_p = int(w*0.3)

    for i in range(size):
        for j in range(size):
            label = labels[i, j]
            if label != 0:  # 数字があるマスのみ描画
                
                cv2.putText(
                    img, str(label), (w_p + w*(j),  h*(i+1)),
                    cv2.FONT_HERSHEY_PLAIN, \
                    4, (0, 0, 255), 6, cv2.LINE_AA)
    return img


def putAns(img, i, j, k, size):
    h, w, c = img.shape
    h = h // size
    w = w // size
    h_p = int(h*0.3)
    w_p = int(w*0.3)

    cv2.putText(img, str(k), (w_p + w*(j),  h*(i+1)), cv2.FONT_HERSHEY_PLAIN, \
        4, (0, 0, 255), 6, cv2.LINE_AA)
