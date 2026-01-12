# config_fixed.py - Cấu hình đầy đủ đã sửa lỗi
import os
from datetime import datetime

# ==================== CAMERA SETTINGS ====================
CAMERA_SOURCES = {
    'primary': 0,
}

CAMERA_AUTO_RECONNECT = True
CAMERA_RECONNECT_DELAY = 3
MAX_CAMERA_RETRIES = 5

AUTO_EXPOSURE = True
AUTO_WHITE_BALANCE = True
AUTO_FOCUS = False

# ==================== DISPLAY SETTINGS ====================
FRAME_WIDTH = 800
FRAME_HEIGHT = 600
DISPLAY_MODE = 'normal'
SHOW_PIP = False

# Màu sắc HUD
COLOR_SAFE = (0, 255, 0)
COLOR_WARNING = (0, 255, 255)
COLOR_DANGER = (0, 0, 255)
COLOR_INFO = (255, 255, 255)

# ==================== ENHANCEMENT SETTINGS ====================
USE_GAMMA = True
GAMMA_VALUE = 1.3
USE_HIST_EQ = False
USE_ADAPTIVE_GAIN = True
MIN_LIGHT_THRESHOLD = 50

# ==================== PERFORMANCE SETTINGS ====================
FPS_BUFFER_SIZE = 20
PERF_UPDATE_INTERVAL = 2.0

# ==================== RECORDING SETTINGS ====================
RECORD_VIDEO = True
RECORD_EVENTS_ONLY = False

VIDEO_OUTPUT_DIR = "recordings"
VIDEO_FPS = 20
VIDEO_FORMAT = "mp4"
VIDEO_CODEC = "mp4v"
EVENT_PRE_BUFFER = 30
EVENT_POST_BUFFER = 30

# Tạo thư mục lưu trữ
os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
VIDEO_OUTPUT = os.path.join(VIDEO_OUTPUT_DIR, f"blackbox_{timestamp}.mp4")

# ==================== ALERT SYSTEM ====================
ENABLE_AUDIO_ALERTS = False
ENABLE_VISUAL_ALERTS = True

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

# ==================== GPS/SENSORS ====================
USE_GPS_SIMULATION = True
SIMULATED_SPEED = 60
SIMULATED_LOCATION = (10.8231, 106.6297)

# ==================== LOGGING SETTINGS ====================
LOG_LEVEL = "INFO"
LOG_DIR = "logs"
LOG_BUFFER_SIZE = 100
LOG_FLUSH_INTERVAL = 5