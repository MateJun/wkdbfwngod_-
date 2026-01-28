"""
YOLO + 스테레오 비전 통합 시스템
- 비동기 처리 (카메라/스테레오/YOLO 별도 스레드)
- 실시간 객체 인식 + 거리 측정
"""

import cv2
import numpy as np
import threading
import time
from ultralytics import YOLO
from kalman_filter import KalmanFilterTracker

# ============================================================
# 설정
# ============================================================
LEFT_CAM = 0
RIGHT_CAM = 2
CALIB_FILE = r"C:\Users\mts20\Desktop\자율주행\calibration_data.npz"

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
TARGET_FPS = 30

YOLO_MODEL = "yolov8n.pt"
YOLO_CONFIDENCE = 0.5
YOLO_IMG_SIZE = 640

DEPTH_MIN = 100    # mm
DEPTH_MAX = 10000  # mm
DEPTH_ROI_SIZE = 40

KALMAN_PROCESS_NOISE = 0.001
KALMAN_MEASUREMENT_NOISE = 1000
KALMAN_TIMEOUT = 90

CLASS_COLORS = {
    "person": (0, 255, 255),
    "car": (0, 255, 0),
    "bus": (0, 165, 255),
    "truck": (0, 128, 255),
    "traffic light": (0, 0, 255),
    "stop sign": (255, 0, 255),
    "default": (255, 255, 255),
}

# ============================================================
# 카메라 스레드
# ============================================================
class CameraThread:
    def __init__(self, camera_id, width, height):
        self.cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError(f"카메라 {camera_id} 열기 실패")

        for _ in range(5):
            ret, frame = self.cap.read()
            if ret and frame is not None:
                print(f"  카메라 {camera_id}: 연결 성공")
                break
        else:
            self.cap.release()
            raise RuntimeError(f"카메라 {camera_id} 프레임 읽기 실패")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

        self.frame = None
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def release(self):
        self.running = False
        self.cap.release()

# ============================================================
# 스테레오 스레드
# ============================================================
class StereoThread:
    def __init__(self, map1_l, map2_l, map1_r, map2_r, Q):
        self.map1_l, self.map2_l = map1_l, map2_l
        self.map1_r, self.map2_r = map1_r, map2_r
        self.Q = Q

        self.stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=128,
            blockSize=9,
            P1=8 * 3 * 81,
            P2=32 * 3 * 81,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32
        )

        self.frame_left = None
        self.frame_right = None
        self.points_3d = None
        self.disparity_visual = None
        self.rect_left = None
        self.rect_right = None

        self.running = True
        self.lock = threading.Lock()
        self.new_frame = threading.Event()
        self.thread = threading.Thread(target=self._process, daemon=True)
        self.thread.start()

    def update_frames(self, left, right):
        with self.lock:
            self.frame_left = left.copy()
            self.frame_right = right.copy()
        self.new_frame.set()

    def _process(self):
        while self.running:
            self.new_frame.wait(timeout=0.1)
            self.new_frame.clear()

            with self.lock:
                if self.frame_left is None or self.frame_right is None:
                    continue
                left = self.frame_left.copy()
                right = self.frame_right.copy()

            rect_left = cv2.remap(left, self.map1_l, self.map2_l, cv2.INTER_LINEAR)
            rect_right = cv2.remap(right, self.map1_r, self.map2_r, cv2.INTER_LINEAR)

            gray_left = cv2.cvtColor(rect_left, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(rect_right, cv2.COLOR_BGR2GRAY)

            disparity = self.stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0
            disp_visual = np.uint8(cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX))
            points = cv2.reprojectImageTo3D(disparity, self.Q)

            with self.lock:
                self.points_3d = points
                self.disparity_visual = disp_visual
                self.rect_left = rect_left
                self.rect_right = rect_right

    def get_points(self):
        with self.lock:
            return self.points_3d.copy() if self.points_3d is not None else None

    def get_display(self):
        with self.lock:
            disp = self.disparity_visual.copy() if self.disparity_visual is not None else None
            left = self.rect_left.copy() if self.rect_left is not None else None
            right = self.rect_right.copy() if self.rect_right is not None else None
        return disp, left, right

    def stop(self):
        self.running = False

# ============================================================
# YOLO 스레드
# ============================================================
class YoloThread:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        import torch
        if torch.cuda.is_available():
            self.model.to('cuda')
            self.device = torch.cuda.get_device_name(0)
        else:
            self.device = "CPU"

        self.frame = None
        self.detections = []

        self.running = True
        self.lock = threading.Lock()
        self.new_frame = threading.Event()
        self.thread = threading.Thread(target=self._process, daemon=True)
        self.thread.start()

    def update_frame(self, frame):
        with self.lock:
            self.frame = frame.copy()
        self.new_frame.set()

    def _process(self):
        while self.running:
            self.new_frame.wait(timeout=0.1)
            self.new_frame.clear()

            with self.lock:
                if self.frame is None:
                    continue
                frame = self.frame.copy()

            results = self.model(frame, verbose=False, imgsz=YOLO_IMG_SIZE)

            detections = []
            for result in results:
                for box in result.boxes:
                    conf = float(box.conf[0])
                    if conf < YOLO_CONFIDENCE:
                        continue
                    cls_id = int(box.cls[0])
                    detections.append({
                        "class": result.names[cls_id],
                        "bbox": tuple(map(int, box.xyxy[0])),
                        "confidence": conf
                    })

            with self.lock:
                self.detections = detections

    def get_detections(self):
        with self.lock:
            return self.detections.copy()

    def stop(self):
        self.running = False

# ============================================================
# 헬퍼 함수
# ============================================================
def get_depth(points_3d, x, y):
    if points_3d is None:
        return None
    h, w = points_3d.shape[:2]
    x1, x2 = max(0, x - DEPTH_ROI_SIZE), min(w, x + DEPTH_ROI_SIZE)
    y1, y2 = max(0, y - DEPTH_ROI_SIZE), min(h, y + DEPTH_ROI_SIZE)
    roi = points_3d[y1:y2, x1:x2, 2]
    valid = roi[(roi > DEPTH_MIN) & (roi < DEPTH_MAX)]
    if len(valid) < 10:
        return None
    q1, q3 = np.percentile(valid, [25, 75])
    iqr = q3 - q1
    filtered = valid[(valid >= q1 - 1.5 * iqr) & (valid <= q3 + 1.5 * iqr)]
    return np.median(filtered) if len(filtered) > 0 else np.median(valid)

def get_color(name):
    return CLASS_COLORS.get(name, CLASS_COLORS["default"])

# ============================================================
# 초기화
# ============================================================
print("=" * 60)
print("YOLO + 스테레오 비전")
print("=" * 60)

print("\n[1/4] 캘리브레이션 로드...")
calib = np.load(CALIB_FILE)
map1_l, map2_l = calib['map1_l'], calib['map2_l']
map1_r, map2_r = calib['map1_r'], calib['map2_r']
Q = calib['Q']
print("  완료")

print("\n[2/4] YOLO 로드...")
yolo = YoloThread(YOLO_MODEL)
print(f"  완료 ({yolo.device})")

print("\n[3/4] 카메라 연결...")
try:
    cam_left = CameraThread(LEFT_CAM, FRAME_WIDTH, FRAME_HEIGHT)
    cam_right = CameraThread(RIGHT_CAM, FRAME_WIDTH, FRAME_HEIGHT)
    time.sleep(1)
    print("  완료")
except RuntimeError as e:
    print(f"  오류: {e}")
    exit(1)

print("\n[4/4] 스테레오 초기화...")
stereo = StereoThread(map1_l, map2_l, map1_r, map2_r, Q)
print("  완료")

kalman = KalmanFilterTracker(KALMAN_PROCESS_NOISE, KALMAN_MEASUREMENT_NOISE, KALMAN_TIMEOUT)

print("\n" + "=" * 60)
print("[q: 종료, s: 캡처 저장]")
print("=" * 60)

# ============================================================
# 메인 루프
# ============================================================
frame_count = 0
fps_time = time.time()
fps = 0

while True:
    t_start = time.time()

    frame_left = cam_left.read()
    frame_right = cam_right.read()
    if frame_left is None or frame_right is None:
        continue

    frame_count += 1
    stereo.update_frames(frame_left, frame_right)
    points_3d = stereo.get_points()
    disp_visual, rect_left, rect_right = stereo.get_display()

    if rect_left is not None:
        yolo.update_frame(rect_left)
        output = rect_left.copy()
    else:
        yolo.update_frame(frame_left)
        output = frame_left.copy()

    detections = yolo.get_detections()

    for det in detections:
        name = det["class"]
        x1, y1, x2, y2 = det["bbox"]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        color = get_color(name)
        depth = get_depth(points_3d, cx, cy)

        if depth and DEPTH_MIN < depth < DEPTH_MAX:
            filtered = kalman.update(name, depth)
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 3)
            cv2.circle(output, (cx, cy), 6, color, -1)

            label = f"{name}: {filtered/1000:.2f}m"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(output, (x1, y1 - lh - 10), (x1 + lw + 10, y1), color, -1)
            cv2.putText(output, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        else:
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            cv2.putText(output, f"{name}: --", (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if frame_count % 10 == 0:
        fps = 10 / (time.time() - fps_time)
        fps_time = time.time()

    cv2.putText(output, f"FPS: {fps:.1f} | Objects: {len(detections)}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    kalman.tick()
    cv2.imshow("YOLO + Stereo", output)

    if rect_right is not None:
        cv2.imshow("Right Camera", rect_right)
    if disp_visual is not None:
        cv2.imshow("Depth Map", disp_visual)

    elapsed = time.time() - t_start
    wait_time = max(1, int((1.0 / TARGET_FPS - elapsed) * 1000))

    key = cv2.waitKey(wait_time) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        cv2.imwrite(f"capture_{frame_count}.jpg", output)
        print(f"저장: capture_{frame_count}.jpg")

# ============================================================
# 종료
# ============================================================
stereo.stop()
yolo.stop()
cam_left.release()
cam_right.release()
cv2.destroyAllWindows()
print("\n종료")
