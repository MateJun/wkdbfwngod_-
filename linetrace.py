"""
[단순 차선 추적 버전] linetrace.py
- Ultra-Fast-Lane-Detection-v2 모델만 사용하여 차선 인식 및 주행
- 목표: 인식된 차선 중 오른쪽에서 두 번째 차선(2차선)을 따라 주행
- 장애물 회피 기능 없음
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
from PIL import ImageFont, ImageDraw, Image

# ======================================================================================
# [1. 튜닝 및 설정 (사용자 수정 구간)]
# ======================================================================================

# ---------------- [1.1. 주행 및 제어 설정] ----------------
PORT = 'COM9'
# ... (이하 동일)
# ---------------- [1.4. 파일 경로] ----------------
UFLD_MODEL = r"C:\Users\mts20\Desktop\자율주행\Ultra-Fast-Lane-Detection-v2\weights\tusimple_res18.pth"
KOREAN_FONT_PATH = "C:/Windows/Fonts/malgun.ttf" # 윈도우 맑은 고딕 폰트 경로


# ======================================================================================
# [2. 시스템 클래스 및 함수]
# ======================================================================================

# ---------------- [2.1. 카메라 처리] ----------------

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
        x = self.relu(self.bn1(self.conv1(x))); x = self.maxpool(x)
        x = self.layer1(x); x2 = self.layer2(x); x3 = self.layer3(x2); x4 = self.layer4(x3)
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
        target_h = UFLD_HEIGHT + UFLD_ROI_TOP_CROP + UFLD_ROI_BOTTOM_CROP
        target_w = UFLD_WIDTH + UFLD_ROI_LEFT_CROP + UFLD_ROI_RIGHT_CROP
        img = cv2.resize(frame, (target_w, target_h))
        y_start, x_start = UFLD_ROI_TOP_CROP, UFLD_ROI_LEFT_CROP
        y_end = -UFLD_ROI_BOTTOM_CROP if UFLD_ROI_BOTTOM_CROP > 0 else target_h
        x_end = -UFLD_ROI_RIGHT_CROP if UFLD_ROI_RIGHT_CROP > 0 else target_w
        img = img[y_start:y_end, x_start:x_end]
        img = cv2.resize(img, (UFLD_WIDTH, UFLD_HEIGHT))
        img = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 - self.mean) / self.std
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

    def get_target_lane_center(self, coords, img_w, img_h):
        if not coords: return None, "차선 없음"
        
        # 화면 중앙 기준 오른쪽 차선들만 필터링
        right_lanes = []
        for lane in coords:
            if not lane: continue
            # 차선 하단부 x좌표를 기준으로 판단
            bottom_x = lane[-1][0]
            if bottom_x > img_w / 2:
                right_lanes.append(lane)
        
        if not right_lanes: return None, "오른쪽 차선 없음"

        # x좌표 기준으로 오른쪽 차선들 정렬 (왼쪽부터 -> 오른쪽으로)
        right_lanes.sort(key=lambda lane: lane[-1][0])

        target_lane = None
        if len(right_lanes) >= 2:
            # 오른쪽 차선이 2개 이상이면, 두 번째 차선(2차선) 선택
            target_lane = right_lanes[1]
            status_msg = "2차선 추적"
        else:
            # 오른쪽 차선이 1개만 있으면, 그 차선(1차선)을 따라감
            target_lane = right_lanes[0]
            status_msg = "1차선 추적"

        # 선택된 차선의 중심 계산 (하단부 기준)
        target_y = int(img_h * 0.9) # 화면 더 아래쪽을 기준으로 중심 계산
        lane_points_y = [p[1] for p in target_lane]
        lane_points_x = [p[0] for p in target_lane]
        
        if not lane_points_y: return None, "차선 포인트 없음"
        
        # target_y에 가장 가까운 점의 x좌표를 차선 중심으로 사용
        center_x = np.interp(target_y, lane_points_y, lane_points_x)
        return int(center_x), status_msg

def draw_lanes(frame, coords, target_center_x):
    # ... (이전과 동일)

def putText_korean(frame, text, pos, font_path, font_size, color):
    """Pillow를 사용하여 OpenCV 이미지에 한글 텍스트를 출력하는 함수"""
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(font_path, font_size)
    draw.text(pos, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# ======================================================================================
# [3. 메인 실행부]
# ======================================================================================

def main():
    # ... (초기화 부분)
    try:
        # ...
        korean_font = ImageFont.truetype(KOREAN_FONT_PATH, 20) # 폰트 객체 미리 로드
    except Exception as e:
        print(f"\n초기화 실패: {e}"); return
    
    # ... (메인 루프)
    while True:
        # ... (키 입력 및 프레임 읽기)
        if is_driving:
            # ... (주행 로직)
            if center_x:
                # ...
                status_text = f"조향값: {steer_val}"
            else:
                # ...
                status_text = lane_info
        
        # --- 화면 출력 (한글 지원) ---
        display_frame = putText_korean(display_frame, f"상태: {status_text}", (10, 20), KOREAN_FONT_PATH, 25, (0, 255, 0))
        
        driving_status_text = f"주행: {'ON' if is_driving else 'OFF'}"
        driving_color = (0, 255, 0) if is_driving else (0, 0, 255)
        display_frame = putText_korean(display_frame, driving_status_text, (10, 60), KOREAN_FONT_PATH, 25, driving_color)

        cv2.imshow("Line Trace Driving", display_frame)

    print("\n프로그램 종료 중...");
    if arduino and arduino.is_open: arduino.write(f"{STOP_SIGNAL}\n".encode()); arduino.close()
    if camera: camera.release()
    cv2.destroyAllWindows(); print("완료")

if __name__ == '__main__':
    try:
        torch_lib_path = os.path.join(os.path.dirname(torch.__file__), 'lib')
        if os.path.exists(torch_lib_path): os.add_dll_directory(torch_lib_path)
    except:
        pass
    os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
    main()
