import numpy as np
import cv2 as cv
import glob
import os
import platform
import subprocess

# Hiển thị thư mục hiện tại
print("📁 Current working directory:", os.getcwd())

# Kích thước bàn cờ: 4x6
CHECKERBOARD = (4, 6)

# Tạo điểm thế giới (tọa độ 3D)
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

objpoints = []
imgpoints = []

images = glob.glob('calib*.jpg')

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        print(f"✅ Tìm thấy bàn cờ trong {fname}")
        objpoints.append(objp)
        imgpoints.append(corners)
    else:
        print(f"❌ Không tìm thấy bàn cờ trong {fname}")

# Hiệu chỉnh camera
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

# In kết quả
print("\n📸 Ma trận nội tại của camera (camera matrix):")
print(mtx)

print("\n🎯 Hệ số biến dạng (distortion coefficients):")
print(dist)

# Lưu vào B.npz
npz_path = os.path.abspath("B.npz")
np.savez(npz_path, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
print(f"💾 Đã lưu file B.npz tại: {npz_path}")

# Lưu vào camera_parameters.txt
txt_path = os.path.abspath("camera_parameters.txt")
with open(txt_path, 'w') as f:
    f.write("Camera Matrix (Intrinsic Parameters):\n")
    f.write(np.array2string(mtx, precision=4, separator=', ') + "\n\n")
    f.write("Distortion Coefficients (k1, k2, p1, p2, k3):\n")
    f.write(np.array2string(dist, precision=4, separator=', ') + "\n")

print(f"📄 Đã lưu camera matrix vào file: {txt_path}")

# Tự động mở file .txt sau khi lưu (Windows, macOS, Linux)
def open_file(filepath):
    if platform.system() == "Windows":
        os.startfile(filepath)
    elif platform.system() == "Darwin":  # macOS
        subprocess.call(["open", filepath])
    elif platform.system() == "Linux":
        subprocess.call(["xdg-open", filepath])

# Gọi hàm mở file
open_file(txt_path)
