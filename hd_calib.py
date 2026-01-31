"""
스테레오 카메라 캘리브레이션 (C920 최적화)
- 720p@60fps (낮은 지연시간)
- 수동 노출/화이트밸런스 (좌우 동기화)
- CV_16SC2 렉티피케이션 맵 (빠른 remap)
"""

import cv2
import numpy as np
import time
import os

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

LEFT_CAM = 1
RIGHT_CAM = 0
CHESSBOARD_SIZE = (8, 6)
SQUARE_SIZE = 30.0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
TARGET_FPS = 60
MIN_CAPTURES = 15
CALIB_FILE = r"C:\Users\mts20\Desktop\자율주행\calibration_data_720p60.npz"

MANUAL_EXPOSURE = -7
MANUAL_WHITE_BALANCE = 4500
MANUAL_GAIN = 100

USE_CUDA = cv2.cuda.getCudaEnabledDeviceCount() > 0

# 전역 카메라 변수 (트랙바 콜백용)
cap_left = None
cap_right = None

def on_exposure(val):
    """노출 트랙바 콜백 (0~12 → -1~-13)"""
    exp = -(val + 1)
    if cap_left and cap_right:
        cap_left.set(cv2.CAP_PROP_EXPOSURE, exp)
        cap_right.set(cv2.CAP_PROP_EXPOSURE, exp)

def on_wb(val):
    """화이트밸런스 트랙바 콜백 (0~37 → 2800~6500)"""
    wb = 2800 + val * 100
    if cap_left and cap_right:
        cap_left.set(cv2.CAP_PROP_WB_TEMPERATURE, wb)
        cap_right.set(cv2.CAP_PROP_WB_TEMPERATURE, wb)

def setup_c920_camera(cap, name):
    # MJPG 코덱 먼저 설정 (60fps 활성화 필수)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_EXPOSURE, MANUAL_EXPOSURE)
    cap.set(cv2.CAP_PROP_WB_TEMPERATURE, MANUAL_WHITE_BALANCE)
    cap.set(cv2.CAP_PROP_GAIN, MANUAL_GAIN)
    cap.set(cv2.CAP_PROP_FOCUS, 0)
    print(f"  {name}: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}@{cap.get(cv2.CAP_PROP_FPS):.0f}fps")

print("=" * 60)
print("스테레오 캘리브레이션 (C920 720p@60fps)")
print("=" * 60)

cap_left = cv2.VideoCapture(LEFT_CAM, cv2.CAP_DSHOW)
cap_right = cv2.VideoCapture(RIGHT_CAM, cv2.CAP_DSHOW)
if not cap_left.isOpened() or not cap_right.isOpened():
    print("카메라 연결 실패!")
    exit(1)

setup_c920_camera(cap_left, "왼쪽")
setup_c920_camera(cap_right, "오른쪽")

for _ in range(30):
    cap_left.read()
    cap_right.read()

if USE_CUDA:
    gpu_frame_l, gpu_frame_r = cv2.cuda_GpuMat(), cv2.cuda_GpuMat()
    gpu_gray_l, gpu_gray_r = cv2.cuda_GpuMat(), cv2.cuda_GpuMat()

objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints, imgpoints_l, imgpoints_r = [], [], []
img_size = None
capture_count = 0
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
fps_counter, fps_start, current_fps = 0, time.time(), 0
frame_count = 0
DETECT_INTERVAL = 5  # 5프레임마다 체스보드 탐지
found_l, found_r = False, False
corners_l, corners_r = None, None

# 윈도우 및 트랙바 설정
cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)

# 노출 트랙바: 0~12 → -1~-13 (기본값 6 → -7)
cv2.createTrackbar("Exposure", "Calibration", 6, 12, on_exposure)

# 화이트밸런스 트랙바: 0~37 → 2800~6500 (기본값 17 → 4500)
cv2.createTrackbar("WB", "Calibration", 17, 37, on_wb)

print("\n[s]캡처 [c]캘리브레이션 [q]종료")
print("트랙바로 노출/WB 조정")

while True:
    ret1, frame_l = cap_left.read()
    ret2, frame_r = cap_right.read()
    if not ret1 or not ret2: continue
    if img_size is None: img_size = (frame_l.shape[1], frame_l.shape[0])

    fps_counter += 1
    if time.time() - fps_start >= 1.0:
        current_fps, fps_counter, fps_start = fps_counter, 0, time.time()

    if USE_CUDA:
        gpu_frame_l.upload(frame_l); gpu_frame_r.upload(frame_r)
        cv2.cuda.cvtColor(gpu_frame_l, cv2.COLOR_BGR2GRAY, gpu_gray_l)
        cv2.cuda.cvtColor(gpu_frame_r, cv2.COLOR_BGR2GRAY, gpu_gray_r)
        gray_l, gray_r = gpu_gray_l.download(), gpu_gray_r.download()
    else:
        gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)

    # 체스보드 탐지 (N프레임마다)
    frame_count += 1
    if frame_count % DETECT_INTERVAL == 0:
        found_l, corners_l = cv2.findChessboardCorners(gray_l, CHESSBOARD_SIZE, flags)
        found_r, corners_r = cv2.findChessboardCorners(gray_r, CHESSBOARD_SIZE, flags)

    disp_l, disp_r = frame_l.copy(), frame_r.copy()
    if found_l and corners_l is not None:
        cv2.drawChessboardCorners(disp_l, CHESSBOARD_SIZE, corners_l, found_l)
    if found_r and corners_r is not None:
        cv2.drawChessboardCorners(disp_r, CHESSBOARD_SIZE, corners_r, found_r)

    cv2.putText(disp_l, f"L:{'OK' if found_l else 'X'}", (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0) if found_l else (0,0,255), 3)
    cv2.putText(disp_r, f"R:{'OK' if found_r else 'X'}", (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0) if found_r else (0,0,255), 3)

    # 현재 설정 표시
    exp_val = int(cap_left.get(cv2.CAP_PROP_EXPOSURE))
    wb_val = int(cap_left.get(cv2.CAP_PROP_WB_TEMPERATURE))
    cv2.putText(disp_l, f"Cnt:{capture_count} FPS:{current_fps}", (20,100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
    cv2.putText(disp_l, f"Exp:{exp_val} WB:{wb_val}K", (20,140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

    cv2.imshow("Calibration", np.hstack([cv2.resize(disp_l,(640,360)), cv2.resize(disp_r,(640,360))]))
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        # 캡처 시 새로 탐지
        found_l, corners_l = cv2.findChessboardCorners(gray_l, CHESSBOARD_SIZE, flags)
        found_r, corners_r = cv2.findChessboardCorners(gray_r, CHESSBOARD_SIZE, flags)
        if found_l and found_r:
            objpoints.append(objp)
            imgpoints_l.append(cv2.cornerSubPix(gray_l, corners_l, (11,11), (-1,-1), criteria))
            imgpoints_r.append(cv2.cornerSubPix(gray_r, corners_r, (11,11), (-1,-1), criteria))
            capture_count += 1
            print(f"캡처 {capture_count}장")
        else:
            print("체스보드 인식 실패")
    elif key == ord('c') and capture_count >= MIN_CAPTURES: break
    elif key == ord('q'): exit(0)

cap_left.release(); cap_right.release(); cv2.destroyAllWindows()

print(f"\n캘리브레이션 ({len(objpoints)}장)...")
ret_l, mtx_l, dist_l, _, _ = cv2.calibrateCamera(objpoints, imgpoints_l, img_size, None, None)
ret_r, mtx_r, dist_r, _, _ = cv2.calibrateCamera(objpoints, imgpoints_r, img_size, None, None)
print(f"Left RMS: {ret_l:.4f}, Right RMS: {ret_r:.4f}")

ret_stereo, mtx_l, dist_l, mtx_r, dist_r, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints_l, imgpoints_r, mtx_l, dist_l, mtx_r, dist_r, img_size,
    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5),
    flags=cv2.CALIB_FIX_INTRINSIC)
print(f"Stereo RMS: {ret_stereo:.4f}")

R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(mtx_l, dist_l, mtx_r, dist_r, img_size, R, T, alpha=0)
map1_l, map2_l = cv2.initUndistortRectifyMap(mtx_l, dist_l, R1, P1, img_size, cv2.CV_16SC2)
map1_r, map2_r = cv2.initUndistortRectifyMap(mtx_r, dist_r, R2, P2, img_size, cv2.CV_16SC2)

np.savez(CALIB_FILE, mtx_l=mtx_l, dist_l=dist_l, mtx_r=mtx_r, dist_r=dist_r,
         R=R, T=T, E=E, F=F, R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
         map1_l=map1_l, map2_l=map2_l, map1_r=map1_r, map2_r=map2_r,
         roi1=roi1, roi2=roi2, img_size=np.array(img_size),
         baseline=np.linalg.norm(T), focal_length=P1[0,0], rms_stereo=ret_stereo)

print(f"\n완료! {CALIB_FILE}")
print(f"Baseline: {np.linalg.norm(T):.2f}mm, 맵타입: CV_16SC2")
input("엔터...")
