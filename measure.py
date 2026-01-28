"""
스테레오 거리 측정
- 마우스 클릭으로 거리 측정
- 흑백 명암비 (가까움=밝음, 멀리=어두움)
"""

import cv2
import numpy as np
import os

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

# ============================================================
# 설정
# ============================================================
LEFT_CAM = 0
RIGHT_CAM = 2
CALIB_FILE = r"C:\Users\mts20\Desktop\자율주행\calibration_data.npz"

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# ============================================================
# 초기화
# ============================================================
print("=" * 50)
print("스테레오 거리 측정")
print("=" * 50)

# 캘리브레이션 로드
if not os.path.exists(CALIB_FILE):
    print(f"캘리브레이션 파일 없음: {CALIB_FILE}")
    exit(1)

calib = np.load(CALIB_FILE)
map1_l, map2_l = calib['map1_l'], calib['map2_l']
map1_r, map2_r = calib['map1_r'], calib['map2_r']
Q = calib['Q']
print("캘리브레이션 로드 완료")

# 카메라
cap_left = cv2.VideoCapture(LEFT_CAM, cv2.CAP_DSHOW)
cap_right = cv2.VideoCapture(RIGHT_CAM, cv2.CAP_DSHOW)

cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

if not cap_left.isOpened() or not cap_right.isOpened():
    print("카메라 연결 실패!")
    exit(1)

print(f"카메라: {FRAME_WIDTH}x{FRAME_HEIGHT}")

# ============================================================
# 변수
# ============================================================
click_point = None
last_measurement = None
measurements = []

# ============================================================
# 콜백
# ============================================================
def mouse_callback(event, x, y, flags, param):
    global click_point
    if event == cv2.EVENT_LBUTTONDOWN:
        click_point = (x, y)

def trackbar_callback(x):
    pass

# ============================================================
# 윈도우 설정
# ============================================================
cv2.namedWindow("Depth Map", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Depth Map", FRAME_WIDTH, FRAME_HEIGHT)
cv2.setMouseCallback("Depth Map", mouse_callback)
cv2.createTrackbar("NumDisp(x16)", "Depth Map", 8, 16, trackbar_callback)
cv2.createTrackbar("BlockSize", "Depth Map", 11, 25, trackbar_callback)

print("\n[조작법]")
print("  마우스 클릭 - 거리 측정")
print("  s - 측정값 기록")
print("  q - 종료")
print("=" * 50)

# ============================================================
# 메인 루프
# ============================================================
while True:
    ret1, frame_left = cap_left.read()
    ret2, frame_right = cap_right.read()

    if not ret1 or not ret2:
        continue

    # Rectification
    rect_left = cv2.remap(frame_left, map1_l, map2_l, cv2.INTER_LINEAR)
    rect_right = cv2.remap(frame_right, map1_r, map2_r, cv2.INTER_LINEAR)

    # Grayscale + Histogram Equalization
    gray_left = cv2.equalizeHist(cv2.cvtColor(rect_left, cv2.COLOR_BGR2GRAY))
    gray_right = cv2.equalizeHist(cv2.cvtColor(rect_right, cv2.COLOR_BGR2GRAY))

    # 슬라이더 값
    num_disp = 16 * max(1, cv2.getTrackbarPos("NumDisp(x16)", "Depth Map"))
    block_size = cv2.getTrackbarPos("BlockSize", "Depth Map")
    block_size = max(5, block_size | 1)  # 홀수, 최소 5

    # 스테레오 매칭
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=8 * 3 * block_size ** 2,
        P2=32 * 3 * block_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=100,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    disparity = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0

    # 시각화
    disp_visual = np.uint8(cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX))
    disp_display = cv2.cvtColor(disp_visual, cv2.COLOR_GRAY2BGR)

    # 클릭 거리 측정
    if click_point:
        x, y = click_point
        if 0 <= x < disparity.shape[1] and 0 <= y < disparity.shape[0]:
            if disparity[y, x] > 0:
                points_3d = cv2.reprojectImageTo3D(disparity, Q)
                depth = points_3d[y, x][2]

                if 0 < depth < 10000:
                    last_measurement = depth
                    cv2.circle(disp_display, (x, y), 8, (0, 255, 0), 2)
                    cv2.line(disp_display, (x-15, y), (x+15, y), (0, 255, 0), 1)
                    cv2.line(disp_display, (x, y-15), (x, y+15), (0, 255, 0), 1)
                    text = f"{depth:.0f}mm ({depth/10:.1f}cm)"
                    cv2.putText(disp_display, text, (x+15, y-15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv2.circle(disp_display, (x, y), 8, (0, 0, 255), 2)
                    cv2.putText(disp_display, "OUT OF RANGE", (x+15, y-15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                cv2.circle(disp_display, (x, y), 8, (0, 0, 255), 2)
                cv2.putText(disp_display, "NO DISPARITY", (x+15, y-15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        click_point = None

    # 정보 표시
    cv2.putText(disp_display, f"NumDisp: {num_disp} | Block: {block_size}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    if last_measurement:
        cv2.putText(disp_display, f"Last: {last_measurement:.0f}mm", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Depth Map", disp_display)
    cv2.imshow("Left Camera", rect_left)

    key = cv2.waitKey(30) & 0xFF

    if key == ord('s') and last_measurement:
        actual = input("실제 거리(mm): ")
        try:
            actual = float(actual)
            error = abs(last_measurement - actual)
            measurements.append({'measured': last_measurement, 'actual': actual, 'error': error})
            print(f"기록: 측정={last_measurement:.0f}mm, 실제={actual:.0f}mm, 오차={error:.0f}mm")
        except:
            print("숫자를 입력하세요")

    elif key == ord('q'):
        break

# ============================================================
# 종료
# ============================================================
cap_left.release()
cap_right.release()
cv2.destroyAllWindows()

if measurements:
    print("\n측정 결과:")
    for i, m in enumerate(measurements, 1):
        print(f"  {i}. 측정={m['measured']:.0f}mm, 실제={m['actual']:.0f}mm, 오차={m['error']:.0f}mm")
    avg_error = np.mean([m['error'] for m in measurements])
    print(f"평균 오차: {avg_error:.0f}mm")
