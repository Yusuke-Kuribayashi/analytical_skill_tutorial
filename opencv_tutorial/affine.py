"""
opencv100本ノック Q28. アフィン変換(平行移動)
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2

from matplot_ssh import is_under_ssh_connection, use_WebAgg

# matplotlibのバックエンド変更(GUI系)
if is_under_ssh_connection():
    use_WebAgg(port = 8100, port_retries = 50)

def affine_transform(image, tx=0, ty=0, angle=0):
    print("アフィン変換を行います")

    # アフィン変換行列の定義
    M = np.float32([[np.cos(angle), -1*np.sin(angle), tx], 
                    [np.sin(angle), np.cos(angle), ty], 
                    [0, 0, 1]])

    # 出力画像の初期化
    output_image = np.zeros_like(image)

    # アフィン変換の実行
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            trans_orig = np.dot(M, np.array([j, i, 1])).astype(int)[:2]

            # 範囲外チェック
            if 0 <= trans_orig[0] < image.shape[1] and 0 <= trans_orig[1] < image.shape[0]:
                output_image[trans_orig[1], trans_orig[0]] = image[i, j]

    return output_image


def main():
    # 画像の読み込み
    origin_image = cv2.imread('dataset/imori.jpg')
    if origin_image is None:
        print("画像が読み込めませんでした。")
        return

    print("オリジナル画像の情報")
    print("shape:", origin_image.shape)
    print("dtype:", origin_image.dtype)

    # アフィン変換の実行
    tx, ty = 30, -30
    angle = np.radians(30)  # 30度の回転
    transformed_image = affine_transform(origin_image, tx, ty, angle)

    # 結果の表示
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(cv2.cvtColor(origin_image, cv2.COLOR_BGR2RGB))
    
    plt.subplot(1, 2, 2)
    plt.title('Transformed Image')
    plt.imshow(cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB))
    
    plt.show()

if __name__ == "__main__":
    main()