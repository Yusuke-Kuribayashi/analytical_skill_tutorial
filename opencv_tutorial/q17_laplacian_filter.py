"""
opencv100本ノック Q.17 ラプラシアンフィルタ
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2

from matplot_ssh import is_under_ssh_connection, use_WebAgg

# matplotlibのバックエンド変更(GUI系)
if is_under_ssh_connection():
    use_WebAgg(port = 8000, port_retries = 50)

# ラプラシアンフィルタの関数
def laplacian_filter_cv(image):
    print("ラプラシアンフィルタを適用します")
    # ラプラシアンフィルタの適用
    laplacian_image = cv2.Laplacian(image, cv2.CV_64F)
    
    # 結果を正規化して表示可能な形式に変換
    laplacian_image = cv2.convertScaleAbs(laplacian_image)
    
    return laplacian_image

def laplacian_filter_scratch(image):
    print("ラプラシアンフィルタを適用します")
    # ラプラシアンフィルタのカーネル
    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]], dtype=np.float32)
    
    # 出力画像の初期化
    laplacian_image = np.zeros_like(image, dtype=np.float32)

    # 入力画像のパディング
    pad_size = 1
    padded_image = cv2.copyMakeBorder(image, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_CONSTANT, value=0)

    # カーネルを適用
    for i in range(pad_size, padded_image.shape[0] - pad_size):
        for j in range(pad_size, padded_image.shape[1] - pad_size):
            # カーネルを適用してラプラシアンを計算
            region = padded_image[i-pad_size:i+pad_size+1, j-pad_size:j+pad_size+1]
            laplacian_value = np.sum(region * kernel)
            laplacian_image[i-pad_size, j-pad_size] = laplacian_value

    laplacian_image = np.clip(laplacian_image, 0, 255).astype(np.uint8)  # 画像の値を0-255にクリップ
    
    return laplacian_image


# メイン関数
def main():
    # 画像の読み込み
    image_path = 'dataset/imori.jpg'  # ここに画像のパスを指定
    origin_image = cv2.imread(image_path)

    if origin_image is None:
        print("画像が読み込めませんでした。パスを確認してください。")
        return
    

    gray_image = cv2.cvtColor(origin_image, cv2.COLOR_BGR2GRAY)


    # ラプラシアンフィルタの適用
    filtered_image_current = laplacian_filter_cv(gray_image)
    filtered_image_scratch = laplacian_filter_scratch(gray_image)

    # 結果の表示
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Current Image')
    plt.imshow(cv2.cvtColor(filtered_image_current, cv2.COLOR_BGR2RGB))
    
    plt.subplot(1, 2, 2)
    plt.title('scratch Image')
    plt.imshow(cv2.cvtColor(filtered_image_scratch, cv2.COLOR_BGR2RGB))
    
    plt.show()

if __name__ == "__main__":
    main()