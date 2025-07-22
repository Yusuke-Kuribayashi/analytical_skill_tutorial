import numpy as np
import matplotlib.pyplot as plt
import cv2

from matplot_ssh import is_under_ssh_connection, use_WebAgg

# matplotlibのバックエンド変更(GUI系)
if is_under_ssh_connection():
    use_WebAgg(port = 8100, port_retries = 50)

def low_pass_filter(image, kernel_size=5):
    print("ローパスフィルタを適用します")
    
    # カーネルの生成
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    
    # フィルタの適用
    filtered_image = cv2.filter2D(image, -1, kernel)
    
    return filtered_image

def main():
    # 画像の読み込み
    origin_image = cv2.imread('dataset/imori.jpg')
    if origin_image is None:
        print("画像が読み込めませんでした。")
        return
    
    gray_image = cv2.cvtColor(origin_image, cv2.COLOR_BGR2GRAY)

    # ローパスフィルタの適用
    filtered_image = low_pass_filter(gray_image, kernel_size=5)

    # 結果の表示
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(cv2.cvtColor(origin_image, cv2.COLOR_BGR2RGB))
    
    plt.subplot(1, 2, 2)
    plt.title('Filtered Image')
    plt.imshow(cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB))
    
    plt.show()

if __name__ == "__main__":
    main()