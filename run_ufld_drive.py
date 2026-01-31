"""
[UFLD 버전] 자율주행 시스템 (Ultra-Fast-Lane-Detection + YOLOv8 장애물 회피)
- 최종 수정: 가독성 향상을 위한 코드 구조 재배치
- 평시: UFLD 기반 차선 추적
- 장애물(차량) 감지 시: 스테레오 카메라 활성화, 거리 측정 및 차선 변경
"""
import os
import sys
import serial
import time
import threading
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision
import onnxruntime as ort
from ultralytics import YOLO
import win32com.client
from pygrabber.dshow_graph import FilterGraph

# ======================================================================================
# [1. 튜닝 및 설정 (사용자 수정 구간)]
# ======================================================================================

# ---------------- [1.1. 주행 및 제어 설정] ----------------
PORT = 'COM9'
BAUD = 9600
SERVO_CENTER = 460
SERVO_MIN = 384
SERVO_MAX = 550
STOP_SIGNAL = 0
AVOID_TARGET_CLASSES = [2, 5, 7]  # YOLO COCO 클래스: car=2, bus=5, truck=7
AVOID_DISTANCE_THRESHOLD = 1500  # mm
LANE_CHANGE_STEER_VAL = SERVO_MIN
LANE_CHANGE_DURATION = 2.0

# ---------------- [1.2. 카메라 및 ROI 설정] ----------------
LEFT_CAM_ID_STRING = r"USB\VID_046D&PID_08E5&MI_00\B&E292FD3&0&0000"
RIGHT_CAM_ID_STRING = r"USB\VID_046D&PID_08E5&MI_00\9&2D2EB393&0&0000"
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
TARGET_FPS = 60
MANUAL_EXPOSURE = -7
MANUAL_WHITE_BALANCE = 4500
UFLD_ROI_TOP_CROP = 0      # UFLD 입력 이미지 상단에서 잘라낼 픽셀 수 (0 = 자르지 않음)
UFLD_ROI_BOTTOM_CROP = 0 # UFLD 입력 이미지 하단에서 잘라낼 픽셀 수 (0 = 자르지 않음)
UFLD_ROI_LEFT_CROP = 0   # UFLD 입력 이미지 왼쪽에서 잘라낼 픽셀 수 (0 = 자르지 않음)
UFLD_ROI_RIGHT_CROP = 0  # UFLD 입력 이미지 오른쪽에서 잘라낼 픽셀 수 (0 = 자르지 않음)

# ---------------- [1.3. AI 모델 파라미터] ----------------
YOLO_CONFIDENCE = 0.5
YOLO_IMG_SIZE = 480
UFLD_WIDTH = 800
UFLD_HEIGHT = 320
UFLD_NUM_ROW = 56
UFLD_NUM_COL = 41
UFLD_NUM_CELL_ROW = 100
UFLD_NUM_CELL_COL = 100
UFLD_NUM_LANES = 4
HITNET_WIDTH = 640
HITNET_HEIGHT = 480

# ---------------- [1.4. 파일 경로] ----------------
CALIB_FILE = r"C:\Users\mts20\Desktop\자율주행\calibration_data_720p60.npz"
YOLO_MODEL = r"C:\Users\mts20\Desktop\자율주행\yolov8n.pt"
HITNET_MODEL = r"C:\Users\mts20\Desktop\자율주행\models\eth3d\saved_model_480x640\model_float32_opt.onnx"
UFLD_MODEL = r"C:\Users\mts20\Desktop\자율주행\Ultra-Fast-Lane-Detection-v2\weights\tusimple_res18.pth"

# ======================================================================================
# [2. 시스템 클래스 및 함수]
# ======================================================================================

# ---------------- [2.1. 카메라 처리] ----------------

def get_camera_indices_by_id(left_cam_id_str, right_cam_id_str):
    print("카메라 인덱스를 고유 ID로 찾는 중...")
    graph = FilterGraph()
    devices = graph.get_input_devices_info()
    left_cam_idx, right_cam_idx = -1, -1

    for i, device_info in enumerate(devices):
        device_path = device_info.get('device_path', '')
        if left_cam_id_str in device_path:
            left_cam_idx = i
            print(f"  - 왼쪽 카메라 찾음: 인덱스 {i}")
        elif right_cam_id_str in device_path:
            right_cam_idx = i
            print(f"  - 오른쪽 카메라 찾음: 인덱스 {i}")

    if left_cam_idx == -1 or right_cam_idx == -1:
        raise RuntimeError("지정된 ID의 카메라를 찾을 수 없습니다. 연결을 확인하세요.")
    return left_cam_idx, right_cam_idx

class CameraThread:
    def __init__(self, cam_id):
        self.cap = cv2.VideoCapture(cam_id, cv2.CAP_MSMF)
        if not self.cap.isOpened(): raise RuntimeError(f"카메라 {cam_id} 열기 실패")
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_WB_TEMPERATURE, MANUAL_WHITE_BALANCE)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, MANUAL_EXPOSURE)
        self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self.cap.set(cv2.CAP_PROP_FOCUS, 0)
        self.frame = None; self.running = True; self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._update, daemon=True); self.thread.start()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock: self.frame = frame
            else: self.running = False
    def read(self):
        with self.lock: return self.frame.copy() if self.frame is not None else None
    def release(self):
        self.running = False; self.cap.release()

# ---------------- [2.2. UFLD 차선 인식] ----------------

class ResNetBackbone(nn.Module):
    def __init__(self, layers='18'):
        super().__init__()
        model = torchvision.models.resnet18(weights=None) if layers == '18' else torchvision.models.resnet34(weights=None)
        self.conv1, self.bn1, self.relu, self.maxpool = model.conv1, model.bn1, model.relu, model.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = model.layer1, model.layer2, model.layer3, model.layer4
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x2, x3, x4

class UFLDNet(nn.Module):
    def __init__(self, backbone='18', **kwargs):
        super().__init__()
        self.model = ResNetBackbone(backbone)
        self.pool = nn.Conv2d(512, 8, 1)
        input_dim = (UFLD_HEIGHT // 32) * (UFLD_WIDTH // 32) * 8
        self.dim1 = UFLD_NUM_CELL_ROW * UFLD_NUM_ROW * UFLD_NUM_LANES
        self.dim2 = UFLD_NUM_CELL_COL * UFLD_NUM_COL * UFLD_NUM_LANES
        self.dim3 = 2 * UFLD_NUM_ROW * UFLD_NUM_LANES
        self.dim4 = 2 * UFLD_NUM_COL * UFLD_NUM_LANES
        total_dim = self.dim1 + self.dim2 + self.dim3 + self.dim4
        self.cls = nn.Sequential(nn.Linear(input_dim, 2048), nn.ReLU(), nn.Linear(2048, total_dim))
    def forward(self, x):
        _, _, fea = self.model(x)
        fea = self.pool(fea)
        fea = fea.view(-1, (UFLD_HEIGHT // 32) * (UFLD_WIDTH // 32) * 8)
        out = self.cls(fea)
        return {
            'loc_row': out[:, :self.dim1].view(-1, UFLD_NUM_CELL_ROW, UFLD_NUM_ROW, UFLD_NUM_LANES),
            'loc_col': out[:, self.dim1:self.dim1+self.dim2].view(-1, UFLD_NUM_CELL_COL, UFLD_NUM_COL, UFLD_NUM_LANES),
            'exist_row': out[:, self.dim1+self.dim2:self.dim1+self.dim2+self.dim3].view(-1, 2, UFLD_NUM_ROW, UFLD_NUM_LANES),
            'exist_col': out[:, -self.dim4:].view(-1, 2, UFLD_NUM_COL, UFLD_NUM_LANES)
        }

class UFLDLaneDetector:
    def __init__(self, model_path):
        print("  UFLD 모델 로딩..."); self.row_anchor = np.linspace(160, 710, UFLD_NUM_ROW) / 720
        self.net = UFLDNet()
        state_dict = torch.load(model_path, map_location='cpu')['model']
        compatible_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        self.net.load_state_dict(compatible_state_dict, strict=False)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.net.to(self.device).eval()
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        print(f"    UFLD 로드 완료 ({self.device.upper()} 모드)")

    def preprocess(self, frame):
        """OpenCV 기반 빠른 전처리. 상하좌우 ROI 크롭 기능 포함."""
        # 1. 크롭을 포함한 전체 필요 사이즈로 리사이즈
        target_h = UFLD_HEIGHT + UFLD_ROI_TOP_CROP + UFLD_ROI_BOTTOM_CROP
        target_w = UFLD_WIDTH + UFLD_ROI_LEFT_CROP + UFLD_ROI_RIGHT_CROP
        img = cv2.resize(frame, (target_w, target_h))

        # 2. 상하좌우 크롭
        y_start = UFLD_ROI_TOP_CROP
        y_end = -UFLD_ROI_BOTTOM_CROP if UFLD_ROI_BOTTOM_CROP > 0 else target_h
        x_start = UFLD_ROI_LEFT_CROP
        x_end = -UFLD_ROI_RIGHT_CROP if UFLD_ROI_RIGHT_CROP > 0 else target_w
        
        img = img[y_start:y_end, x_start:x_end]

        # 3. 최종 크기 보정
        img = cv2.resize(img, (UFLD_WIDTH, UFLD_HEIGHT))
        
        # 4. BGR -> RGB, 정규화
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        return torch.from_numpy(img.transpose(2, 0, 1)[np.newaxis, ...]).float()

    def detect(self, frame):
        img_tensor = self.preprocess(frame).to(self.device)
        with torch.no_grad(): pred = self.net(img_tensor)
        return self.pred2coords(pred, frame.shape[1], frame.shape[0])

    def pred2coords(self, pred, img_w, img_h):
        max_indices_row = pred['loc_row'].argmax(1).cpu()
        valid_row = pred['exist_row'].argmax(1).cpu()
        coords = []
        for i in range(UFLD_NUM_LANES):
            tmp = []
            if valid_row[0, :, i].sum() > UFLD_NUM_ROW / 4:
                for k in range(valid_row.shape[1]):
                    if valid_row[0, k, i]:
                        all_ind = torch.arange(max(0, max_indices_row[0, k, i] - 1), min(UFLD_NUM_CELL_ROW - 1, max_indices_row[0, k, i] + 1) + 1)
                        out_tmp = (pred['loc_row'][0, all_ind, k, i].softmax(0).cpu() * all_ind.float()).sum() + 0.5
                        out_tmp = out_tmp / (UFLD_NUM_CELL_ROW - 1) * img_w
                        tmp.append((int(out_tmp), int(self.row_anchor[k] * img_h)))
                coords.append(tmp)
        return coords

    def get_lane_center(self, coords, img_w, img_h):
        if not coords: return None, "No lanes detected"
        target_y = int(img_h * 0.8)
        left_x, right_x = None, None
        for lane in coords:
            if len(lane) < 2: continue
            closest_pt = min(lane, key=lambda p: abs(p[1] - target_y))
            x, y = closest_pt
            if x < img_w / 2: left_x = max(left_x, x) if left_x is not None else x
            else: right_x = min(right_x, x) if right_x is not None else x
        if left_x and right_x: return (left_x + right_x) // 2, f"L:{left_x} R:{right_x}"
        if right_x: return right_x - 200, f"R only:{right_x}"
        if left_x: return left_x + 200, f"L only:{left_x}"
        return None, "Center calculation failed"

def draw_lanes(frame, coords):
    for lane in coords:
        for pt in lane: cv2.circle(frame, pt, 5, (0, 255, 0), -1)
    return frame

# ---------------- [2.3. 스테레오 거리 측정] ----------------

def get_depth_from_bbox(depth_map, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    roi = depth_map[y1 + (y2 - y1) // 4:y2 - (y2 - y1) // 4, x1 + (x2 - x1) // 4:x2 - (x2 - x1) // 4]
    if roi.size == 0: return None
    valid = roi[(roi > 100) & (roi < 30000)]
    return np.median(valid) if len(valid) > 10 else None

# ---------------- [2.4. 주행 로직] ----------------

def drive_logic_frame(cams, models, calib_data, arduino, drive_state):
    rect_left = cv2.remap(cams['left'].read(), calib_data['map1_l'], calib_data['map2_l'], cv2.INTER_LINEAR)
    rect_right = cv2.remap(cams['right'].read(), calib_data['map1_r'], calib_data['map2_r'], cv2.INTER_LINEAR)
    display_frame = rect_right.copy()
    status_text = ""

    if drive_state['DRIVING_STATE'] == "LANE_FOLLOW":
        coords = models['ufld'].detect(rect_right)
        center_x, lane_info = models['ufld'].get_lane_center(coords, FRAME_WIDTH, FRAME_HEIGHT)
        display_frame = draw_lanes(display_frame, coords)
        if center_x:
            steer_val = int(np.interp(center_x, [0, FRAME_WIDTH], [SERVO_MIN, SERVO_MAX]))
            if arduino: arduino.write(f"{steer_val}\n".encode())
            status_text = f"Steer: {steer_val}"
            cv2.circle(display_frame, (center_x, int(FRAME_HEIGHT * 0.8)), 10, (0, 255, 255), -1)
        else:
            if arduino: arduino.write(f"{SERVO_CENTER}\n".encode())
            status_text = lane_info
        results = models['yolo'](rect_right, verbose=False, imgsz=YOLO_IMG_SIZE, classes=AVOID_TARGET_CLASSES)
        if len(results[0].boxes) > 0 and results[0].boxes[0].conf > YOLO_CONFIDENCE:
            drive_state['DRIVING_STATE'] = "STEREO_AVOIDANCE"; print("장애물 감지! 스테레오 모드 전환.")

    elif drive_state['DRIVING_STATE'] == "STEREO_AVOIDANCE":
        status_text = "Stereo: Measuring Distance"
        inp = np.stack([cv2.cvtColor(cv2.resize(rect_left, (HITNET_WIDTH, HITNET_HEIGHT)), cv2.COLOR_BGR2GRAY),
                        cv2.cvtColor(cv2.resize(rect_right, (HITNET_WIDTH, HITNET_HEIGHT)), cv2.COLOR_BGR2GRAY)], axis=0)[np.newaxis, ...].astype(np.float32) / 255.0
        disparity = models['hitnet'].run(None, {models['hitnet_input']: inp})[0].squeeze()
        disparity_resized = cv2.resize(disparity, (FRAME_WIDTH, FRAME_HEIGHT)) * (FRAME_WIDTH / HITNET_WIDTH)
        depth_map = (calib_data['focal_length'] * calib_data['baseline']) / (disparity_resized + 1e-6)
        results = models['yolo'](rect_right, verbose=False, imgsz=YOLO_IMG_SIZE, classes=AVOID_TARGET_CLASSES)
        avoid_triggered = False
        if len(results[0].boxes) > 0 and results[0].boxes[0].conf > YOLO_CONFIDENCE:
            dist = get_depth_from_bbox(depth_map, results[0].boxes[0].xyxy[0])
            if dist and dist < AVOID_DISTANCE_THRESHOLD:
                print(f"!!! {dist/1000:.1f}m 앞 장애물! 차선 변경 시작 !!!")
                if arduino: arduino.write(f"{LANE_CHANGE_STEER_VAL}\n".encode())
                drive_state['DRIVING_STATE'] = "LANE_CHANGING"
                drive_state['lane_change_finish_time'] = time.time() + LANE_CHANGE_DURATION
                avoid_triggered = True
        if not avoid_triggered: drive_state['DRIVING_STATE'] = "LANE_FOLLOW"; print("장애물 안전 거리. 차선 유지 모드 복귀.")

    elif drive_state['DRIVING_STATE'] == "LANE_CHANGING":
        status_text = f"LANE CHANGING! ({drive_state['lane_change_finish_time'] - time.time():.1f}s left)"
        if arduino: arduino.write(f"{LANE_CHANGE_STEER_VAL}\n".encode())
        if time.time() > drive_state['lane_change_finish_time']:
            drive_state['DRIVING_STATE'] = "LANE_FOLLOW"; print("차선 변경 완료. 차선 유지 모드 복귀.")

    cv2.putText(display_frame, f"STATE: {drive_state['DRIVING_STATE']}", (10, 30), 1, 1.5, (255,255,0), 2)
    cv2.putText(display_frame, f"CMD: {status_text}", (10, 70), 1, 1.5, (255,255,0), 2)
    cv2.imshow("UFLD Driving View", display_frame)

# ======================================================================================
# [3. 메인 실행부]
# ======================================================================================

def main():
    print("="*60 + "\nUFLD 자율주행 시스템 시작\n" + "="*60)
    arduino, cams, models, calib_data = None, {}, {}, {}
    try:
        # --- 장치 및 모델 초기화 ---
        print("\n[1/5] 아두이노 연결..."); arduino = serial.Serial(PORT, BAUD, timeout=1); time.sleep(2); print("  완료")
        print("\n[2/5] AI 모델 로드...")
        models['ufld'] = UFLDLaneDetector(UFLD_MODEL)
        providers = ort.get_available_providers()
        models['hitnet'] = ort.InferenceSession(HITNET_MODEL, providers=['CUDAExecutionProvider' if 'CUDAExecutionProvider' in providers else 'CPUExecutionProvider'])
        models['hitnet_input'] = models['hitnet'].get_inputs()[0].name
        models['yolo'] = YOLO(YOLO_MODEL)
        if torch.cuda.is_available(): models['yolo'].to('cuda')
        print("  완료")
        print("\n[3/5] 캘리브레이션 로드..."); calib = np.load(CALIB_FILE)
        calib_data = {k: v for k, v in calib.items()}; calib_data.update({'focal_length': float(calib['Q'][2,3]), 'baseline': float(abs(1/calib['Q'][3,2]))}); print("  완료")
        print("\n[4/5] 카메라 인덱스 탐색..."); left_idx, right_idx = get_camera_indices_by_id(LEFT_CAM_ID_STRING, RIGHT_CAM_ID_STRING); print("  완료")
        print("\n[5/5] 카메라 연결..."); cams['left'] = CameraThread(left_idx); cams['right'] = CameraThread(right_idx); time.sleep(1); print("  완료")
    except Exception as e: print(f"\n초기화 실패: {e}"); import traceback; traceback.print_exc(); return

    # --- 주행 시작 ---
    drive_state = {'DRIVING_STATE': "LANE_FOLLOW", 'lane_change_finish_time': 0, 'is_driving': False}
    print("\n" + "="*60 + "\ns: 주행 시작/정지 | q: 종료\n" + "="*60)
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'): drive_state['is_driving'] = not drive_state['is_driving']; print(f"주행: {'ON' if drive_state['is_driving'] else 'OFF'}")
        elif key == ord('q'): break
        if not drive_state['is_driving']:
            if arduino: arduino.write(f"{STOP_SIGNAL}\n".encode())
            frame = cams['right'].read()
            if frame is not None: cv2.imshow("UFLD Driving View", frame)
            continue
        try: drive_logic_frame(cams, models, calib_data, arduino, drive_state)
        except Exception as e: print(f"주행 로직 에러: {e}"); import traceback; traceback.print_exc(); break

    # --- 종료 처리 ---
    print("\n프로그램 종료 중..."); 
    if arduino and arduino.is_open: arduino.write(f"{STOP_SIGNAL}\n".encode()); arduino.close()
    for cam in cams.values(): cam.release()
    cv2.destroyAllWindows(); print("완료")

if __name__ == '__main__':
    # 환경 설정
    try:
        torch_lib_path = os.path.join(os.path.dirname(torch.__file__), 'lib')
        if os.path.exists(torch_lib_path): os.add_dll_directory(torch_lib_path)
    except: pass
    os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
    main()
