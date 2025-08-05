#モデル判定の後、機械的に0と判定する。
#ノイズに合わせて推定結果を書き換える
def systematic_pred_zero(split, pred): #by aoki
    for i in range(len(split)):
        #28x28用の処理
        center = split[i][9:19, 9:19, :]  # 中央の10×10を取り出す
        #entire = split[i][:, :, :]    #画像全体を取り出す

        # 黒、白と判定するしきい値（2値化されているため黒は0, 白は1）
        black_threshold = 0
        #white_threshold = 1

        # 黒、白の割合のしきい値
        ratio_b = 0.80
        #ratio_w = 0.07

        # 黒ピクセルの割合を計算(中央の黒の割合が80%以上ならば推定結果を0に書き換える)
        black_ratio = np.mean(center <= black_threshold)
        # print(black_ratio)
        if black_ratio >= ratio_b:
            pred[i] = [1,0,0,0,0,0,0,0,0,0]

        # 白ピクセルの割合を計算(全体の白の割合が7%以下ならば推定結果を0に書き換える)
        # white_ratio = np.mean(entire >= white_threshold)
        # print(white_ratio)
        # if white_ratio <= ratio_w:
        #    pred[i] = [1,0,0,0,0,0,0,0,0,0]
    
    return pred

#端の線を除外 中央10x10に着目するにはradius=5にする。円ではない
#途切れる場合があるので、2,3ピクセルだったら同じ島とみなすなど変更の検討
#by gemini 即席なので内部はあまり見れていない
def extract_center_object(binary_image, radius= 7): #by gemini
    """
    28x28用に作成した。
    二値化画像の中央領域に存在する白ピクセルからたどり、大きいものを抽出
    その他の関数の重心移動で、中央に持ってくる
    
    """
    # 1. 画像内のすべての連結成分（白ピクセルの島）をラベリングする
    # num_labels: 見つかったラベルの総数（背景含む）
    # labels: 各ピクセルにラベル番号を割り当てた画像
    # stats: 各ラベルの統計情報（[左端X, 上端Y, 幅, 高さ, 面積]）
    # centroids: 各ラベルの重心
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)

    # 画像の高さと幅を取得
    h, w = binary_image.shape

    # 2. 中央の領域を定義
    center_x, center_y = w // 2, h // 2
    x_start, x_end = max(0, center_x - radius), min(w, center_x + radius)
    y_start, y_end = max(0, center_y - radius), min(h, center_y + radius)
    
    # 中央領域に存在するラベルのユニークなリストを取得
    center_labels = np.unique(labels[y_start:y_end, x_start:x_end])

    # 3. 中央領域で最大の面積を持つオブジェクトのラベルを見つける
    target_label = -1
    max_area = -1
    
    for label in center_labels:
        # ラベル0は背景なのでスキップ
        if label == 0:
            continue
        
        #除外ロジック--------------------------------------
        #端の枠線が入っている場合(1と間違うので消しておく)
        # オブジェクトのバウンディングボックス情報を取得
        left = stats[label, cv2.CC_STAT_LEFT]
        top = stats[label, cv2.CC_STAT_TOP]
        width = stats[label, cv2.CC_STAT_WIDTH]
        height = stats[label, cv2.CC_STAT_HEIGHT]
        
        # 条件1: オブジェクトが画像の上端から下端まで伸びているか
        is_vertical_line = (top == 0 and (top + height) >= h * 0.9)
        
        # 条件2: オブジェクトが画像の左端から右端まで伸びているか
        is_horizontal_line = (left == 0 and (left + width) >= w * 0.9)
        
        # 縦線または横線と判断された場合は、このラベルをスキップして次の候補へ
        if is_vertical_line or is_horizontal_line:
            continue
        #--------------------------------------------------
        
        # statsからラベルの面積を取得
        area = stats[label, cv2.CC_STAT_AREA]
        
        if area > max_area:
            max_area = area
            target_label = label

    # 4. 新しい画像を生成し、対象のオブジェクトだけを描画
    output_image = np.zeros_like(binary_image)
    if target_label != -1:
        # labels画像内でtarget_labelに一致するピクセルだけを白(255)にする
        output_image[labels == target_label] = 255
        
    return output_image

#閾値の範囲の島を結合する関数。extract_center_objectでは途切れてしまうため
#by gemini 即席なので内部はあまり見れていない
def extract_center_object_distant(binary_image, radius=7, distance_threshold=2): 
    """
    二値化画像から、中央の主要な連結成分のみを抽出する。
    最初に枠線状のオブジェクトを除外し、残った島同士をグループ化して扱う。

    Args:
        binary_image (np.ndarray): 入力となる二値化画像（白: 255, 黒: 0）。
        radius (int, optional): オブジェクトを検出する中央領域の半径。
        distance_threshold (int, optional): この距離以下の隙間にある島同士を同じグループと見なす。

    Returns:
        np.ndarray: 中央のオブジェクトのみが描画された新しい二値化画像。
    """
    # 1. 画像内のすべての連結成分（白ピクセルの島）をラベリングする
    # num_labels: 見つかったラベルの総数（背景含む）
    # labels: 各ピクセルにラベル番号を割り当てた画像
    # stats: 各ラベルの統計情報（[左端X, 上端Y, 幅, 高さ, 面積]）
    # centroids: 各ラベルの重心
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, connectivity=8)

    if num_labels <= 1:
        return np.zeros_like(binary_image)

    h, w = binary_image.shape
    
    # 2. 枠線状の島を除外し、有効なラベルのセットを作成する (事前フィルタリング)
    valid_labels = set()
    for label in range(1, num_labels):
        # オブジェクトのバウンディングボックス情報を取得
        left = stats[label, cv2.CC_STAT_LEFT]
        top = stats[label, cv2.CC_STAT_TOP]
        width = stats[label, cv2.CC_STAT_WIDTH]
        height = stats[label, cv2.CC_STAT_HEIGHT]
        
        #枠線除去
        # 条件1: オブジェクトが画像の上端から下端まで伸びているか
        is_vertical_line = (top == 0 and (top + height) >= h * 0.9)
        # 条件2: オブジェクトが画像の左端から右端まで伸びているか
        is_horizontal_line = (left == 0 and (left + width) >= w * 0.9)
        
        #枠の角の除外を行う。
        #
        # is_right_up_vertex = (left == 0 and top > (h // 2))
        
        # 縦線でも横線でもない場合、有効なラベルとして追加
        if not (is_vertical_line or is_horizontal_line):
            valid_labels.add(label)

    # 3. 有効な島同士をグループ化する (Disjoint Set Union)
    parent = list(range(num_labels))
    def find_set(i):
        if parent[i] == i: return i
        parent[i] = find_set(parent[i])
        return parent[i]
    def unite_sets(i, j):
        i, j = find_set(i), find_set(j)
        if i != j: parent[j] = i

    # 有効なラベルのペアに対してのみ距離計算とグループ化を行う
    valid_labels_list = list(valid_labels)
    for i in range(len(valid_labels_list)):
        for j in range(i + 1, len(valid_labels_list)):
            label1 = valid_labels_list[i]
            label2 = valid_labels_list[j]

            xi, yi, wi, hi = stats[label1, cv2.CC_STAT_LEFT:cv2.CC_STAT_LEFT+4]
            xj, yj, wj, hj = stats[label2, cv2.CC_STAT_LEFT:cv2.CC_STAT_LEFT+4]
            
            dist_x = max(0, max(xi, xj) - min(xi + wi, xj + wj))
            dist_y = max(0, max(yi, yj) - min(yi + hi, yj + hj))
            distance = np.sqrt(dist_x**2 + dist_y**2)
            
            if distance <= distance_threshold:
                unite_sets(label1, label2)

    # 4. グループごとに情報を集計する
    groups = {}
    for label in valid_labels: # 有効なラベルのみを対象
        root = find_set(label)
        if root not in groups:
            groups[root] = {'labels': [], 'area': 0}
        
        groups[root]['labels'].append(label)
        groups[root]['area'] += stats[label, cv2.CC_STAT_AREA]
        
    # 5. 中央領域に存在するラベルから、最適なグループを見つける
    center_x, center_y = w // 2, h // 2
    x_start, x_end = max(0, center_x - radius), min(w, center_x + radius)
    y_start, y_end = max(0, center_y - radius), min(h, center_y + radius)
    center_labels_on_image = np.unique(labels[y_start:y_end, x_start:x_end])

    target_group_root = -1
    max_area = -1

    for label in center_labels_on_image:
        if label not in valid_labels: # 枠線など、無効なラベルは無視
            continue
        
        root = find_set(label)
        group = groups[root]
        
        if group['area'] > max_area:
            max_area = group['area']
            target_group_root = root

    # 6. 新しい画像を生成し、対象グループの全島を描画
    output_image = np.zeros_like(binary_image)
    if target_group_root != -1:
        for label_to_draw in groups[target_group_root]['labels']:
            output_image[labels == label_to_draw] = 255
            
    return output_image



#抽出した数字を中央に配置する by gemini
def center_object_by_centroid(binary_image):
    """
    二値化画像に含まれる白いオブジェクトの重心が、画像の中心に来るように移動させる。

    Args:
        binary_image (np.ndarray): 白いオブジェクトが1つ含まれる二値化画像。

    Returns:
        np.ndarray: オブジェクトが中央に配置された新しい画像。
    """
    # 1. 画像のモーメントを計算
    # 第二引数をTrueにすることで、0以外のピクセルを1として扱う
    moments = cv2.moments(binary_image, True)

    # 2. 重心を計算 (m00は面積)
    # 面積が0の場合（画像が真っ黒の場合）はエラーを避ける
    if moments["m00"] == 0:
        return binary_image # 何もせず元の画像を返す

    centroid_x = int(moments["m10"] / moments["m00"])
    centroid_y = int(moments["m01"] / moments["m00"])

    # 3. 画像の中心座標と移動量を計算
    h, w = binary_image.shape
    image_center_x = w // 2
    image_center_y = h // 2
    
    shift_x = image_center_x - centroid_x
    shift_y = image_center_y - centroid_y

    # 4. 平行移動のためのアフィン変換行列を作成
    M = np.float32([
        [1, 0, shift_x],
        [0, 1, shift_y]
    ])

    # 5. アフィン変換を適用して画像を平行移動
    centered_image = cv2.warpAffine(binary_image, M, (w, h))

    return centered_image