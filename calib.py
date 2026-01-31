"""
스테레오 카메라 캘리브레이션
- 체스보드: 내부 코너 8x6 (9x7 칸)
- CUDA 가속 사용 (그레이스케일 변환)
"""

import cv2
import numpy as np
import time
import os

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

# ============================================================
# 설정
# ============================================================
LEFT_CAM = 0
RIGHT_CAM = 1

CHESSBOARD_SIZE = (8, 6)  # 내부 코너 (가로 9칸-1, 세로 7칸-1)
SQUARE_SIZE = 30.0        # mm

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

MIN_CAPTURES = 10         # 최소 캡처 수

CALIB_FILE = r"C:\Users\mts20\Desktop\자율주행\calibration_data.npz"

# ============================================================
# CUDA 확인
# ============================================================
USE_CUDA = cv2.cuda.getCudaEnabledDeviceCount() > 0

# ============================================================
# 초기화
# ============================================================
print("=" * 50)
print("스테레오 카메라 캘리브레이션")
print("=" * 50)
print(f"CUDA: {USE_CUDA}")
print(f"내부 코너: {CHESSBOARD_SIZE[0]}x{CHESSBOARD_SIZE[1]} = {CHESSBOARD_SIZE[0]*CHESSBOARD_SIZE[1]}점")
print(f"사각형 크기: {SQUARE_SIZE}mm")

# ============================================================
# 카메라 연결
# ============================================================
print("\n[1/2] 카메라 연결...")
cap_left = cv2.VideoCapture(LEFT_CAM, cv2.CAP_MSMF)
cap_right = cv2.VideoCapture(RIGHT_CAM, cv2.CAP_MSMF)

if not cap_left.isOpened() or not cap_right.isOpened():
    print("  카메라 연결 실패!")
    exit(1)

cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap_left.set(cv2.CAP_PROP_BUFFERSIZE, 1)

cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap_right.set(cv2.CAP_PROP_BUFFERSIZE, 1)

print(f"  왼쪽: {int(cap_left.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap_left.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
print(f"  오른쪽: {int(cap_right.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap_right.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

# ============================================================
# CUDA 메모리 (재사용)
# ============================================================
if USE_CUDA:
    gpu_frame_l = cv2.cuda_GpuMat()
    gpu_frame_r = cv2.cuda_GpuMat()
    gpu_gray_l = cv2.cuda_GpuMat()
    gpu_gray_r = cv2.cuda_GpuMat()

# ============================================================
# 변수 초기화
# ============================================================
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []
imgpoints_l = []
imgpoints_r = []
img_size = None
capture_count = 0

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE

found_l, found_r = False, False
corners_l, corners_r = None, None

# ============================================================
# 윈도우 설정
# ============================================================
cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)

print("\n" + "=" * 50)
print("[조작법]")
print("  s - 캡처")
print(f"  c - 캘리브레이션 ({MIN_CAPTURES}장 이상)")
print("  q - 종료")
print("=" * 50)

# ============================================================
# 캡처 루프
# ============================================================
while True:
    ret1, frame_l = cap_left.read()
    ret2, frame_r = cap_right.read()

    if not ret1 or not ret2:
        continue

    if img_size is None:
        img_size = (frame_l.shape[1], frame_l.shape[0])

    # 그레이스케일 변환
    if USE_CUDA:
        gpu_frame_l.upload(frame_l)
        gpu_frame_r.upload(frame_r)
        cv2.cuda.cvtColor(gpu_frame_l, cv2.COLOR_BGR2GRAY, gpu_gray_l)
        cv2.cuda.cvtColor(gpu_frame_r, cv2.COLOR_BGR2GRAY, gpu_gray_r)
        gray_l = gpu_gray_l.download()
        gray_r = gpu_gray_r.download()
    else:
        gray_l = cv2.cvtColor(frame_l, cv2.COLOR_BGR2GRAY)
        gray_r = cv2.cvtColor(frame_r, cv2.COLOR_BGR2GRAY)

    # 체스보드 찾기
    found_l, corners_l = cv2.findChessboardCorners(gray_l, CHESSBOARD_SIZE, flags)
    found_r, corners_r = cv2.findChessboardCorners(gray_r, CHESSBOARD_SIZE, flags)

    # 화면 표시
    disp_l = frame_l.copy()
    disp_r = frame_r.copy()

    if found_l and corners_l is not None:
        cv2.drawChessboardCorners(disp_l, CHESSBOARD_SIZE, corners_l, found_l)
        cv2.putText(disp_l, "L: OK", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    else:
        cv2.putText(disp_l, "L: X", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    if found_r and corners_r is not None:
        cv2.drawChessboardCorners(disp_r, CHESSBOARD_SIZE, corners_r, found_r)
        cv2.putText(disp_r, "R: OK", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    else:
        cv2.putText(disp_r, "R: X", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    cv2.putText(disp_l, f"Count: {capture_count}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    combined = np.hstack([cv2.resize(disp_l, (640, 360)), cv2.resize(disp_r, (640, 360))])
    cv2.imshow("Calibration", combined)

    # 키 입력 처리
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        if found_l and found_r:
            corners_l_ref = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
            corners_r_ref = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)

            objpoints.append(objp)
            imgpoints_l.append(corners_l_ref)
            imgpoints_r.append(corners_r_ref)
            capture_count += 1
            print(f"캡처 {capture_count}장")
        else:
            print("양쪽 모두 인식 필요!")

    elif key == ord('c'):
        if capture_count >= MIN_CAPTURES:
            print(f"\n캘리브레이션 시작 ({capture_count}장)...")
            break
        else:
            print(f"최소 {MIN_CAPTURES}장 필요 (현재 {capture_count}장)")

    elif key == ord('q'):
        cap_left.release()
        cap_right.release()
        cv2.destroyAllWindows()
        exit(0)

# ============================================================
# 카메라 종료
# ============================================================
cap_left.release()
cap_right.release()
cv2.destroyAllWindows()

# ============================================================
# 캘리브레이션 실행
# ============================================================
try:
    print("\n" + "=" * 50)
    print(f"캘리브레이션 ({len(objpoints)}장, {img_size})")
    print("=" * 50)

    # 왼쪽 카메라
    print("\n[1/4] 왼쪽 카메라 캘리브레이션...")
    t1 = time.time()
    ret_l, mtx_l, dist_l, _, _ = cv2.calibrateCamera(objpoints, imgpoints_l, img_size, None, None)
    print(f"  RMS: {ret_l:.4f} ({time.time()-t1:.1f}초)")

    # 오른쪽 카메라
    print("\n[2/4] 오른쪽 카메라 캘리브레이션...")
    t2 = time.time()
    ret_r, mtx_r, dist_r, _, _ = cv2.calibrateCamera(objpoints, imgpoints_r, img_size, None, None)
    print(f"  RMS: {ret_r:.4f} ({time.time()-t2:.1f}초)")

    # 스테레오 캘리브레이션
    print("\n[3/4] 스테레오 캘리브레이션...")
    t3 = time.time()
    flags_stereo = cv2.CALIB_FIX_INTRINSIC
    criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

    ret_stereo, mtx_l, dist_l, mtx_r, dist_r, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_l, imgpoints_r,
        mtx_l, dist_l, mtx_r, dist_r,
        img_size, criteria=criteria_stereo, flags=flags_stereo
    )
    print(f"  RMS: {ret_stereo:.4f} ({time.time()-t3:.1f}초)")

    # 스테레오 정합
    print("\n[4/4] 스테레오 정합...")
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        mtx_l, dist_l, mtx_r, dist_r, img_size, R, T,
        alpha=1, flags=cv2.CALIB_ZERO_DISPARITY
    )

    map1_l, map2_l = cv2.initUndistortRectifyMap(mtx_l, dist_l, R1, P1, img_size, cv2.CV_32FC1)
    map1_r, map2_r = cv2.initUndistortRectifyMap(mtx_r, dist_r, R2, P2, img_size, cv2.CV_32FC1)

    # 저장
    np.savez(CALIB_FILE,
             mtx_l=mtx_l, dist_l=dist_l,
             mtx_r=mtx_r, dist_r=dist_r,
             R=R, T=T, E=E, F=F,
             R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
             map1_l=map1_l, map2_l=map2_l,
             map1_r=map1_r, map2_r=map2_r,
             roi1=roi1, roi2=roi2)

    # 결과 출력
    print("\n" + "=" * 50)
    print("캘리브레이션 완료")
    print("=" * 50)
    print(f"파일: {CALIB_FILE}")
    print(f"Baseline: {np.linalg.norm(T):.2f}mm")

except Exception as e:
    print(f"\n에러: {e}")
    import traceback
    traceback.print_exc()
    input("\n엔터를 누르면 종료...")
    exit(1)

input("\n엔터를 누르면 종료...")
