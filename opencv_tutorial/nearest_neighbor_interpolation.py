import numpy as np
import matplotlib.pyplot as plt
import cv2

from matplot_ssh import is_under_ssh_connection, use_WebAgg

# matplotlibのバックエンド変更(GUI系)
if is_under_ssh_connection():
    use_WebAgg(port = 8100, port_retries = 50)

# 最近傍補完を行う関数
def nearest_neighbor_interpolation(image, scale_factor):
    print("最近傍補完を行います")
    height, width = image.shape[:2]
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)

    # 出力画像の初期化
    output_image = np.zeros((new_height, new_width, 3), dtype=image.dtype)

    for i in range(new_height):
        for j in range(new_width):
            # 元画像の対応するピクセル位置を計算
            orig_x = int(j / scale_factor)
            orig_y = int(i / scale_factor)

            # 範囲外チェック
            orig_x = min(orig_x, width - 1)
            orig_y = min(orig_y, height - 1)

            # 最近傍のピクセル値をコピー
            output_image[i, j] = image[orig_y, orig_x]

    return output_image

def bi_linear_interpolation(image, scale_factor):
    print("バイリニア補完を行います")
    height, width = image.shape[:2]
    new_height = int(height * scale_factor)
    new_width = int(width * scale_factor)

    # 出力画像の初期化
    output_image = np.zeros((new_height, new_width, 3), dtype=image.dtype)

    for i in range(new_height):
        for j in range(new_width):
            # 元画像の対応するピクセル位置を計算
            orig_x = j / scale_factor
            orig_y = i / scale_factor

            # 範囲外チェック
            x1 = min(int(orig_x), width - 1)
            y1 = min(int(orig_y), height - 1)
            x2 = min(x1 + 1, width - 1)
            y2 = min(y1 + 1, height - 1)

            # ピクセル値の取得
            Q11 = image[y1, x1]
            Q12 = image[y2, x1]
            Q21 = image[y1, x2]
            Q22 = image[y2, x2]

            # 補間計算
            dx = orig_x - x1
            dy = orig_y - y1
            output_image[i, j] = (
                Q11 * (1-dx) * (1-dy) +
                Q21 * (1-dx) * (dy) +
                Q12 * (dx)   * (1-dy) +
                Q22 * (dx)   * (dy)
            )

    return output_image

def main():
    # 画像の読み込み
    origin_image = cv2.imread("dataset/imori.jpg")
    if origin_image is None:
        print("画像が読み込めませんでした。")
        return

    # 最近傍補完を適用
    scale_factor = 1.5  # 拡大率
    nn_resized_image = nearest_neighbor_interpolation(origin_image, scale_factor)
    bl_resized_image = nearest_neighbor_interpolation(origin_image, scale_factor)


    # 結果の表示
    plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # plt.title('Original Image')
    # plt.imshow(cv2.cvtColor(origin_image, cv2.COLOR_BGR2RGB), extent=[0, origin_image.shape[1], 0, origin_image.shape[0]])
    # plt.axis("off")

    
    plt.subplot(1, 2, 1)
    plt.title('Resized Image (Nearest Neighbor)')
    plt.imshow(cv2.cvtColor(nn_resized_image, cv2.COLOR_BGR2RGB), extent=[0, nn_resized_image.shape[1], 0, nn_resized_image.shape[0]])
    plt.axis("off")


    plt.subplot(1, 2, 2)
    plt.title('Resized Image (Nearest Neighbor)')
    plt.imshow(cv2.cvtColor(bl_resized_image, cv2.COLOR_BGR2RGB), extent=[0, bl_resized_image.shape[1], 0, bl_resized_image.shape[0]])
    plt.axis("off")

    
    plt.show()

if __name__ == "__main__":
    main()