# config.py - Cấu hình nâng cao
import os
from datetime import datetime

# ==================== CAMERA SETTINGS ====================
CAMERA_SOURCES = {
    'primary': 0,           # Camera chính (0 = webcam)
    'secondary': 1,         # Camera phụ (nếu có)
    'ip_camera': None,      # Địa chỉ RTSP/HTTP
}

CAMERA_AUTO_RECONNECT = True
CAMERA_RECONNECT_DELAY = 3  # giây
MAX_CAMERA_RETRIES = 5

# Tự động điều chỉnh
AUTO_EXPOSURE = True
AUTO_WHITE_BALANCE = True
AUTO_FOCUS = False

# ==================== DISPLAY SETTINGS ====================
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
DISPLAY_MODE = 'normal'     # normal/minimal/expert/pip
SHOW_PIP = True             # Picture-in-Picture
PIP_SIZE = (320, 240)       # Kích thước PIP

# Màu sắc HUD
COLOR_SAFE = (0, 255, 0)    # Xanh lá - an toàn
COLOR_WARNING = (0, 255, 255) # Vàng - cảnh báo
COLOR_DANGER = (0, 0, 255)   # Đỏ - nguy hiểm
COLOR_INFO = (255, 255, 255) # Trắng - thông tin

# ==================== ENHANCEMENT SETTINGS ====================
USE_GAMMA = True
GAMMA_VALUE = 1.3
USE_HIST_EQ = False
USE_ADAPTIVE_GAIN = True    # Tự động điều chỉnh gain theo ánh sáng
MIN_LIGHT_THRESHOLD = 30    # Ngưỡng ánh sáng tối thiểu

# ==================== PERFORMANCE SETTINGS ====================
FPS_BUFFER_SIZE = 30
DYNAMIC_RESOLUTION = True   # Tự động scale resolution
MIN_RESOLUTION = (640, 480)
USE_MULTITHREADING = True
THREAD_POOL_SIZE = 4

# ==================== RECORDING SETTINGS ====================
RECORD_VIDEO = True
RECORD_EVENTS_ONLY = False  # Chỉ ghi khi có sự kiện
EVENT_PRE_BUFFER = 30       # Frame ghi trước khi event
EVENT_POST_BUFFER = 60      # Frame ghi sau khi event

VIDEO_OUTPUT_DIR = "recordings"
VIDEO_FPS = 30
VIDEO_FORMAT = "mp4"
VIDEO_CODEC = "avc1"

# Tạo thư mục lưu trữ
os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
VIDEO_OUTPUT = os.path.join(VIDEO_OUTPUT_DIR, f"blackbox_{timestamp}.{VIDEO_FORMAT}")

# ==================== ALERT SYSTEM ====================
ENABLE_AUDIO_ALERTS = True
ENABLE_VISUAL_ALERTS = True
ALERT_LEVELS = {
    'low': {'color': COLOR_WARNING, 'sound_freq': 440, 'duration': 0.5},
    'medium': {'color': COLOR_WARNING, 'sound_freq': 660, 'duration': 1.0},
    'high': {'color': COLOR_DANGER, 'sound_freq': 880, 'duration': 2.0},
}

# ==================== LOGGING ====================
ENABLE_LOGGING = True
LOG_LEVEL = "INFO"  # DEBUG/INFO/WARNING/ERROR
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = f"system_log_{timestamp}.log"

# ==================== GPS/SENSORS ====================
USE_GPS_SIMULATION = True   # Giả lập GPS nếu không có thật
SIMULATED_SPEED = 60        # km/h
SIMULATED_LOCATION = (10.8231, 106.6297)  # Tọa độ mẫu

# ==================== HOTKEYS ====================
HOTKEYS = {
    'quit': ord('q'),
    'snapshot': ord('s'),
    'toggle_record': ord('r'),
    'toggle_pip': ord('p'),
    'cycle_display': ord('d'),
    'toggle_enhancement': ord('e'),
    'calibrate': ord('c'),
    'show_help': ord('h'),
}