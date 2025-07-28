"""
opencv100本ノック Q5. hsv変換
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2

from matplot_ssh import is_under_ssh_connection, use_WebAgg

# matplotlibのバックエンド変更(GUI系)
if is_under_ssh_connection():
    use_WebAgg(port = 8000, port_retries = 50)


# RGBからHSVへの変換
def BGRtoHSV(origin_image):
    print("RGBをHSVに変換します")
    origin_image = origin_image.astype(np.float32) / 255.0  # 0-255 -> 0-1に変換
    # 出力されるHSV画像の初期化
    output_image = np.zeros_like(origin_image, dtype=np.float32)

    # 各ピクセルごとに、RGBの最大・最小を求める
    Max = np.max(origin_image.copy(), axis=2)
    Min = np.min(origin_image.copy(), axis=2)

    # RGBの取得
    B = origin_image[:, :, 0]
    G = origin_image[:, :, 1]
    R = origin_image[:, :, 2]

    rgb_diff = Max - Min
    for i in range(output_image.shape[0]):
        for j in range(output_image.shape[1]):
            if rgb_diff[i, j] == 0:
                output_image[i, j, 0] = 0
            # Bが最小の場合
            elif Min[i, j] == B[i,j]:
                output_image[i, j, 0] = 60 * (G[i,j] - R[i,j]) / rgb_diff[i, j] + 60
            # Rが最小の場合     
            elif Min[i, j] == R[i,j]:
                output_image[i, j, 0] = 60 * (B[i,j] - G[i,j]) / rgb_diff[i, j] + 180
            # Gが最小の場合
            elif Min[i, j] == G[i,j]:
                output_image[i, j, 0] = 60 * (R[i,j] - B[i,j]) / rgb_diff[i, j] + 300
                
            output_image[i, j, 1] = rgb_diff[i, j]
            output_image[i, j, 2] = Max[i, j]

    return output_image

# HSVからRGBへの変換
def HSVtoRGB(origin_image):
    print("HSVをRGBに変換します")
    output_image = np.zeros_like(origin_image, dtype=np.float32)


    # HSVの取得
    # H = origin_image[:, :, 0].astype(np.float32) / 179.0 * 360.0  # H: 0–360
    H = origin_image[:, :, 0] = (origin_image[:, :, 0] + 180) % 360
    S = origin_image[:, :, 1].astype(np.float32) #/ 255.0          # S: 0–1
    V = origin_image[:, :, 2].astype(np.float32) #/ 255.0          # V: 0–1


    # RGBの取得
    C = S
    H_dash = H / 60.0
    X = C * (1- np.abs(H_dash % 2 - 1))

    z = np.zeros_like(output_image)
    for i in range(H_dash.shape[0]):
        for j in range(H_dash.shape[1]):
            if 0 <= H_dash[i, j] < 1:
                z[i, j] = [C[i,j], X[i,j], 0]
            elif 1 <= H_dash[i, j] < 2:
                z[i, j] = [X[i,j], C[i,j], 0]
            elif 2 <= H_dash[i, j] < 3:
                z[i, j] = [0, C[i,j], X[i,j]]
            elif 3 <= H_dash[i, j] < 4:
                z[i, j] = [0, X[i,j], C[i,j]]
            elif 4 <= H_dash[i, j] < 5:
                z[i, j] = [X[i,j], 0, C[i,j]]
            elif 5 <= H_dash[i, j] < 6:
                z[i, j] = [C[i,j], 0, X[i,j]]

    m = (V-C)
    output_image = np.stack([m,m,m], axis=-1) + z

    # print(output_image.max(), output_image.min())

    output_image = (np.clip(output_image, 0, 1) * 255).astype(np.uint8)  # 0-1にクリップ

    return output_image 

# 解答 ####
def BGR2HSV(_img):
	img = _img.copy() / 255.

	hsv = np.zeros_like(img, dtype=np.float32)

	# get max and min
	max_v = np.max(img, axis=2).copy()
	min_v = np.min(img, axis=2).copy()
	min_arg = np.argmin(img, axis=2)

	# H
	hsv[..., 0][np.where(max_v == min_v)]= 0
	## if min == B
	ind = np.where(min_arg == 0)
	hsv[..., 0][ind] = 60 * (img[..., 1][ind] - img[..., 2][ind]) / (max_v[ind] - min_v[ind]) + 60
	## if min == R
	ind = np.where(min_arg == 2)
	hsv[..., 0][ind] = 60 * (img[..., 0][ind] - img[..., 1][ind]) / (max_v[ind] - min_v[ind]) + 180
	## if min == G
	ind = np.where(min_arg == 1)
	hsv[..., 0][ind] = 60 * (img[..., 2][ind] - img[..., 0][ind]) / (max_v[ind] - min_v[ind]) + 300
		
	# S
	hsv[..., 1] = max_v.copy() - min_v.copy()

	# V
	hsv[..., 2] = max_v.copy()
	
	return hsv

def HSV2BGR(_img, hsv):
	img = _img.copy() / 255.

	# get max and min
	max_v = np.max(img, axis=2).copy()
	min_v = np.min(img, axis=2).copy()

	out = np.zeros_like(img)

	H = hsv[..., 0]
	S = hsv[..., 1]
	V = hsv[..., 2]

	C = S
	H_ = H / 60.
	X = C * (1 - np.abs( H_ % 2 - 1))
	Z = np.zeros_like(H)

	vals = [[Z,X,C], [Z,C,X], [X,C,Z], [C,X,Z], [C,Z,X], [X,Z,C]]

	for i in range(6):
		ind = np.where((i <= H_) & (H_ < (i+1)))
		out[..., 0][ind] = (V - C)[ind] + vals[i][0][ind]
		out[..., 1][ind] = (V - C)[ind] + vals[i][1][ind]
		out[..., 2][ind] = (V - C)[ind] + vals[i][2][ind]

	out[np.where(max_v == min_v)] = 0
	out = np.clip(out, 0, 1)
	out = (out * 255).astype(np.uint8)

	return out
##########

origin_image = cv2.imread("dataset/imori.jpg")
if origin_image is None:
    print("画像が読み込めません")
    exit()

# オリジナル画像の情報
print("オリジナル画像の情報")
print("shape:", origin_image.shape)
print("dtype:", origin_image.dtype)

# opencvパッケージの使用
# 1. RGB->HSV変換
image_hsv = cv2.cvtColor(origin_image, cv2.COLOR_BGR2HSV)
# 2. HSVのH値を180反転
image_hsv[:, :, 0] = (image_hsv[:, :, 0] + 90) % 180
# 3. HSV->BGR変換
currect_image = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)
currect_image = cv2.cvtColor(currect_image, cv2.COLOR_BGR2RGB)
######################

# 解答 ###############
hsv_current = BGR2HSV(origin_image.astype(np.float32))
hsv_current[..., 0] = (hsv_current[..., 0] + 180) % 360
current2_image = cv2.cvtColor(HSV2BGR(origin_image.astype(np.float32), hsv_current), cv2.COLOR_BGR2RGB)
######################

# RGBからHSVへの変換 ##
hsv_image = BGRtoHSV(origin_image.copy())
convert_rgb = HSVtoRGB(hsv_image)
######################


# 2枚並べて表示
fig, axs = plt.subplots(1, 3, figsize=(10, 6))
axs[0].imshow(currect_image)
axs[0].set_title("opencv package image")
axs[1].imshow(current2_image)
axs[1].set_title("current_image")
axs[2].imshow(convert_rgb)
axs[2].set_title("my convert image")

for ax in axs:
    ax.axis("off")  # 軸を非表示

plt.tight_layout()
plt.show()


