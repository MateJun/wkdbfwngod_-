"""
칼만 필터
- 거리 측정값 노이즈 제거 및 안정화
"""

# ============================================================
# 설정
# ============================================================
DEFAULT_PROCESS_NOISE = 0.01
DEFAULT_MEASUREMENT_NOISE = 10
DEFAULT_TIMEOUT = 30

# ============================================================
# 1D 칼만 필터
# ============================================================
class KalmanFilter1D:
    def __init__(self, process_noise=DEFAULT_PROCESS_NOISE, measurement_noise=DEFAULT_MEASUREMENT_NOISE):
        self.Q = process_noise
        self.R = measurement_noise
        self.x = None
        self.P = 1.0
        self.initialized = False

    def update(self, measurement):
        if measurement is None or measurement <= 0:
            return self.x if self.initialized else 0

        if not self.initialized:
            self.x = measurement
            self.initialized = True
            return self.x

        # Predict
        x_pred = self.x
        P_pred = self.P + self.Q

        # Update
        K = P_pred / (P_pred + self.R)
        self.x = x_pred + K * (measurement - x_pred)
        self.P = (1 - K) * P_pred

        return self.x

    def reset(self):
        self.x = None
        self.P = 1.0
        self.initialized = False

# ============================================================
# 멀티 객체 트래커
# ============================================================
class KalmanFilterTracker:
    def __init__(self, process_noise=DEFAULT_PROCESS_NOISE, measurement_noise=DEFAULT_MEASUREMENT_NOISE, timeout=DEFAULT_TIMEOUT):
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.timeout = timeout

        self.filters = {}
        self.last_update = {}
        self.frame_count = 0

    def update(self, object_id, measurement):
        if object_id not in self.filters:
            self.filters[object_id] = KalmanFilter1D(self.process_noise, self.measurement_noise)

        filtered = self.filters[object_id].update(measurement)
        self.last_update[object_id] = self.frame_count
        return filtered

    def tick(self):
        self.frame_count += 1

        # 타임아웃된 객체 제거
        expired = [k for k, v in self.last_update.items() if self.frame_count - v > self.timeout]
        for k in expired:
            del self.filters[k]
            del self.last_update[k]

    def reset_all(self):
        self.filters.clear()
        self.last_update.clear()
        self.frame_count = 0
