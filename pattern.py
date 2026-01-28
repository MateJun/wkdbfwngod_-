"""
체스보드 패턴 생성 (인쇄용)
"""

import cv2
import numpy as np

# ============================================================
# 설정
# ============================================================
SQUARES_X = 9   # 가로 칸 수
SQUARES_Y = 7   # 세로 칸 수
SQUARE_SIZE_PX = 300  # 픽셀 단위

OUTPUT_FILE = "chessboard_print.png"

# ============================================================
# 생성
# ============================================================
width = SQUARES_X * SQUARE_SIZE_PX
height = SQUARES_Y * SQUARE_SIZE_PX

chessboard = np.zeros((height, width), dtype=np.uint8)

for i in range(SQUARES_Y):
    for j in range(SQUARES_X):
        if (i + j) % 2 == 0:
            y1, y2 = i * SQUARE_SIZE_PX, (i + 1) * SQUARE_SIZE_PX
            x1, x2 = j * SQUARE_SIZE_PX, (j + 1) * SQUARE_SIZE_PX
            chessboard[y1:y2, x1:x2] = 255

cv2.imwrite(OUTPUT_FILE, chessboard)

# ============================================================
# 출력
# ============================================================
print("=" * 50)
print("체스보드 생성 완료")
print("=" * 50)
print(f"파일: {OUTPUT_FILE}")
print(f"크기: {width}x{height} 픽셀")
print(f"칸 수: {SQUARES_X}x{SQUARES_Y}")
print(f"내부 코너: {SQUARES_X-1}x{SQUARES_Y-1} = {(SQUARES_X-1)*(SQUARES_Y-1)}점")
print()
print("[인쇄 방법]")
print("1. A3 용지에 인쇄")
print("2. 자로 한 칸 크기 측정 (mm)")
print("3. calib.py의 SQUARE_SIZE 수정")
