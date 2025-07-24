"""
opencv100本ノック Q33. フーリエ変換ローパスフィルタ
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2

from matplot_ssh import is_under_ssh_connection, use_WebAgg

# matplotlibのバックエンド変更(GUI系)
if is_under_ssh_connection():
    use_WebAgg(port = 8100, port_retries = 50)

def low_pass_filter(image):
    print("ローパスフィルタを適用します")

    # フーリエ変換により、周波数空間に変換
    spectrum_image = dft2d(image)
    # 画像の中心を移動
    spectrum_image = transform_image(spectrum_image)  

    # カーネルの生成
    kernel = np.zeros_like(image, dtype=np.float32)
    cv2.circle(kernel, (image.shape[1] // 2, image.shape[0] // 2), image.shape[0] // 4, 1, -1)

    # 畳み込み演算
    low_pass_image = spectrum_image * kernel

    # 画像をもとに戻す
    low_pass_image = transform_image(low_pass_image)
    # 逆フーリエ変換により、空間空間に戻す
    filtered_image = idft2d(low_pass_image)
    # filtered_image = idft(low_pass_image)

    return filtered_image, spectrum_image, kernel

def dft2d(image):
    H, W = image.shape
    x = np.arange(H).reshape(H, 1)
    y = np.arange(W).reshape(W, 1)
    u = np.arange(H).reshape(1, H)
    v = np.arange(W).reshape(1, W)

    # 2次元の離散フーリエ変換の計算
    exp_x = np.exp(-2j * np.pi * x @ u / H)
    exp_y = np.exp(-2j * np.pi * y @ v / W)

    return exp_x @ image @ exp_y.T

def idft2d(f_image):
    H, W = f_image.shape
    x = np.arange(H).reshape(H, 1)
    y = np.arange(W).reshape(W, 1)
    u = np.arange(H).reshape(1, H)
    v = np.arange(W).reshape(1, W)

    # 2次元の逆離散フーリエ変換の計算
    exp_x = np.exp(2j * np.pi * x @ u / H)
    exp_y = np.exp(2j * np.pi * y @ v / W)

    output = np.real(exp_x @ f_image @ exp_y.T / (H * W))
    output = (output - output.min()) / (output.max() - output.min()) * 255  # 正規化

    return output.astype(np.uint8)

def spectrum_visualization(filtered_image):
    magnitude = np.abs(filtered_image)           # 振幅を取得（√(Re² + Im²)）
    magnitude_log = np.log(magnitude + 1)        # 見やすくするために log スケールへ

    return magnitude_log

def transform_image(image):
    H, W = image.shape[:2]
    output = np.zeros_like(image)
    output[H//2:, W//2:] = image[:H//2, :W//2]   # 左上
    output[H//2:, :W//2] = image[:H//2, W//2:]   # 右上
    output[:H//2, :W//2] = image[H//2:, W//2:]  # 右下
    output[:H//2, W//2:] = image[H//2:, :W//2]   # 左下

    return output

# 解答 ######
# DFT hyper-parameters
K, L = 128, 128
channel = 3

# DFT
def dft(img):
	# Prepare DFT coefficient
	G = np.zeros((L, K, channel), dtype=np.complex64)
	H, W, C = img.shape

	# prepare processed index corresponding to original image positions
	x = np.tile(np.arange(W), (H, 1))
	y = np.arange(H).repeat(W).reshape(H, -1)

	# dft
	for c in range(channel):
		for l in range(L):
			for k in range(K):
				G[l, k, c] = np.sum(img[..., c] * np.exp(-2j * np.pi * (x * k / K + y * l / L))) / np.sqrt(K * L)

	return G

# IDFT
def idft(G):
	# prepare out image
	H, W, _ = G.shape
	out = np.zeros((H, W, channel), dtype=np.float32)

	# prepare processed index corresponding to original image positions
	x = np.tile(np.arange(W), (H, 1))
	y = np.arange(H).repeat(W).reshape(H, -1)

	# idft
	for c in range(channel):
		for l in range(H):
			for k in range(W):
				out[l, k, c] = np.abs(np.sum(G[..., c] * np.exp(2j * np.pi * (x * k / W + y * l / H)))) / np.sqrt(W * H)

	# clipping
	out = np.clip(out, 0, 255)
	out = out.astype(np.uint8)

	return out

# LPF
def lpf(G, ratio=0.5):
	H, W, _ = G.shape	

	# transfer positions
	_G = np.zeros_like(G)
	_G[:H//2, :W//2] = G[H//2:, W//2:]
	_G[:H//2, W//2:] = G[H//2:, :W//2]
	_G[H//2:, :W//2] = G[:H//2, W//2:]
	_G[H//2:, W//2:] = G[:H//2, :W//2]

	# get distance from center (H / 2, W / 2)
	x = np.tile(np.arange(W), (H, 1))
	y = np.arange(H).repeat(W).reshape(H, -1)

	# make filter
	_x = x - W // 2
	_y = y - H // 2
	r = np.sqrt(_x ** 2 + _y ** 2)
	mask = np.ones((H, W), dtype=np.float32)
	mask[r > (W // 2 * ratio)] = 0

	mask = np.repeat(mask, channel).reshape(H, W, channel)

	# filtering
	_G *= mask

	# reverse original positions
	G[:H//2, :W//2] = _G[H//2:, W//2:]
	G[:H//2, W//2:] = _G[H//2:, :W//2]
	G[H//2:, :W//2] = _G[:H//2, W//2:]
	G[H//2:, W//2:] = _G[:H//2, :W//2]

	return G
#############

def main():
    # 画像の読み込み
    origin_image = cv2.imread('dataset/imori.jpg')
    if origin_image is None:
        print("画像が読み込めませんでした。")
        return
    
    gray_image = cv2.cvtColor(origin_image, cv2.COLOR_BGR2GRAY)

    # # ローパスフィルタの適用
    filtered_image, spectrum_image, kernel = low_pass_filter(gray_image)
    magnitude_shifted = spectrum_visualization(spectrum_image)

    # DFT
    G = dft(origin_image)
    # LPF
    G = lpf(G)
    # IDFT
    out = idft(G)

    # print(filtered_image.max(), filtered_image.min())

    # 結果の表示
    plt.subplot(1, 3, 1)
    plt.title('Original Image')
    plt.imshow(gray_image, cmap='gray')
    # plt.imshow(out)
    
    plt.subplot(1, 3, 2)
    plt.title('Filtered Image')
    plt.imshow(filtered_image, cmap='gray')
    
    plt.subplot(1, 3, 3)
    plt.title('spectrum Image')
    # plt.imshow(magnitude_shifted, cmap='gray')
    plt.imshow(out, cmap='gray')

    
    plt.show()

if __name__ == "__main__":
    main()