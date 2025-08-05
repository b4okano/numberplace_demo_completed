import cv2
import numpy as np
from ._detect_corners import detect_corners

BASE_SIZE = 64
def trim_board(img, corners):
    w = BASE_SIZE * 14
    h = BASE_SIZE * 15
    transform = cv2.getPerspectiveTransform(np.float32(corners), np.float32([[0, 0], [w, 0], [w, h], [0, h]]))
    normed = cv2.warpPerspective(img, transform, (w, h))
    return normed

def expand_corners(corners, expand_ratio=0.05):
    """
    四隅の座標を画像中心方向から外側に移動させる。
    expand_ratio: 拡大率。0.05なら5%だけ広げる
    """
    # 四隅の重心を中心として使う
    center = np.mean(corners, axis=0)

    # 各頂点をベクトルとして中心から外側にスケーリング
    expanded = []
    for pt in corners:
        vec = pt - center
        new_pt = center + vec * (1 + expand_ratio)
        expanded.append(new_pt)
    
    return np.round(np.array(expanded)).astype(int)
