# core/camera.py - Nâng cấp đa nguồn và tự động kết nối
import cv2
import time
import threading
from enum import Enum
import numpy as np

class CameraType(Enum):
    WEBCAM = "webcam"
    IP_CAMERA = "ip"
    VIDEO_FILE = "video"

class Camera:
    def __init__(self, source_config, width=1280, height=720, camera_id="primary"):
        """
        Khởi tạo camera với cấu hình nâng cao
        
        Args:
            source_config: Cấu hình nguồn (int, str, hoặc dict)
            width: Chiều rộng frame
            height: Chiều cao frame
            camera_id: ID camera (primary/secondary)
        """
        self.camera_id = camera_id
        self.source_config = source_config
        self.width = width
        self.height = height
        self.cap = None
        self.connected = False
        self.retry_count = 0
        self.max_retries = 5
        self.reconnect_delay = 3
        self.connection_lock = threading.Lock()
        
        # Thông tin camera
        self.camera_info = {
            'type': None,
            'fps': 0,
            'resolution': (width, height),
            'properties': {}
        }
        
        # Auto-adjustment
        self.auto_exposure = True
        self.auto_white_balance = True
        self.auto_focus = False
        
        # Statistics
        self.frame_count = 0
        self.drop_count = 0
        self.last_frame_time = 0
        
        # Buffer cho PIP
        self.frame_buffer = None
        
        self._initialize_camera()
    
    def _initialize_camera(self):
        """Khởi tạo và cấu hình camera"""
        with self.connection_lock:
            try:
                # Xác định loại camera
                if isinstance(self.source_config, int):
                    self.camera_info['type'] = CameraType.WEBCAM
                    source = self.source_config
                elif isinstance(self.source_config, str):
                    if self.source_config.startswith(('rtsp://', 'http://')):
                        self.camera_info['type'] = CameraType.IP_CAMERA
                        # Thêm buffer cho camera IP
                        source = self.source_config
                    else:
                        self.camera_info['type'] = CameraType.VIDEO_FILE
                        source = self.source_config
                else:
                    raise ValueError("Invalid camera source configuration")
                
                # Mở camera với timeout
                self.cap = cv2.VideoCapture(source)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Giảm buffer để giảm latency
                
                # Đặt timeout cho camera IP
                if self.camera_info['type'] == CameraType.IP_CAMERA:
                    self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
                
                # Kiểm tra kết nối
                if not self.cap.isOpened():
                    raise ConnectionError(f"Không thể kết nối camera {self.camera_id}")
                
                # Cấu hình camera properties
                self._configure_camera()
                
                # Lấy thông tin camera
                self._get_camera_info()
                
                self.connected = True
                self.retry_count = 0
                print(f"✅ Camera {self.camera_id} ({self.camera_info['type'].value}) đã kết nối")
                
            except Exception as e:
                print(f"❌ Lỗi khởi tạo camera {self.camera_id}: {e}")
                self.connected = False
                self._cleanup()
    
    def _configure_camera(self):
        """Cấu hình thông số camera"""
        if self.cap is None:
            return
        
        # Đặt resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        # Auto settings
        if self.auto_exposure:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # Auto exposure
        else:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Manual exposure
            
        if self.auto_white_balance:
            self.cap.set(cv2.CAP_PROP_AUTO_WB, 1)
            
        if self.auto_focus:
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        
        # Ưu tiên chất lượng hơn tốc độ
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    
    def _get_camera_info(self):
        """Lấy thông tin camera"""
        if self.cap is None:
            return
        
        try:
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.camera_info['resolution'] = (actual_width, actual_height)
            
            self.camera_info['fps'] = self.cap.get(cv2.CAP_PROP_FPS)
            if self.camera_info['fps'] <= 0:
                self.camera_info['fps'] = 30  # Default
            
            # Lấy các properties khác
            props = {
                'brightness': self.cap.get(cv2.CAP_PROP_BRIGHTNESS),
                'contrast': self.cap.get(cv2.CAP_PROP_CONTRAST),
                'saturation': self.cap.get(cv2.CAP_PROP_SATURATION),
                'hue': self.cap.get(cv2.CAP_PROP_HUE),
                'gain': self.cap.get(cv2.CAP_PROP_GAIN),
                'exposure': self.cap.get(cv2.CAP_PROP_EXPOSURE),
            }
            self.camera_info['properties'] = props
            
        except Exception as e:
            print(f"⚠️ Không thể lấy thông tin camera: {e}")
    
    def _check_connection(self):
        """Kiểm tra kết nối camera"""
        if self.cap is None:
            return False
        
        # Đọc thử một frame
        test_frame = self._read_frame_with_timeout()
        if test_frame is not None:
            self.frame_buffer = test_frame
            return True
        
        return False
    
    def _read_frame_with_timeout(self, timeout=1000):
        """Đọc frame với timeout"""
        if self.cap is None:
            return None
        
        # Thử đọc frame
        ret, frame = self.cap.read()
        if ret and frame is not None:
            return frame
        
        return None
    
    def read(self, retry_on_fail=True):
        """
        Đọc frame từ camera với khả năng retry
        
        Returns:
            frame: Frame đã đọc hoặc None nếu lỗi
            status: Trạng thái đọc
        """
        if not self.connected and retry_on_fail:
            self._reconnect()
        
        frame = None
        try:
            with self.connection_lock:
                if self.cap is not None and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    
                    if ret and frame is not None:
                        # Resize về kích thước mong muốn
                        if frame.shape[1] != self.width or frame.shape[0] != self.height:
                            frame = cv2.resize(frame, (self.width, self.height))
                        
                        self.frame_count += 1
                        self.frame_buffer = frame.copy()
                        self.last_frame_time = time.time()
                        
                        return frame, True
                    else:
                        self.drop_count += 1
                        
                        # Sử dụng frame buffer nếu có
                        if self.frame_buffer is not None:
                            print(f"⚠️ Camera {self.camera_id}: Sử dụng frame buffer")
                            return self.frame_buffer.copy(), False
                        
        except Exception as e:
            print(f"❌ Lỗi đọc frame từ camera {self.camera_id}: {e}")
        
        # Thử reconnect nếu cần
        if retry_on_fail and self._should_reconnect():
            self._reconnect()
            # Thử đọc lại sau khi reconnect
            if self.connected:
                return self.read(retry_on_fail=False)
        
        return None, False
    
    def _should_reconnect(self):
        """Kiểm tra có nên reconnect không"""
        current_time = time.time()
        if self.last_frame_time == 0:
            return True
        
        # Nếu quá 2 giây không có frame mới
        if current_time - self.last_frame_time > 2.0:
            return True
        
        # Nếu drop rate quá cao (>30%)
        if self.frame_count > 10 and self.drop_count / self.frame_count > 0.3:
            return True
        
        return False
    
    def _reconnect(self):
        """Tự động kết nối lại camera"""
        if self.retry_count >= self.max_retries:
            print(f"⚠️ Đã thử kết nối lại {self.max_retries} lần, dừng thử")
            return
        
        self.retry_count += 1
        print(f"🔄 Thử kết nối lại camera {self.camera_id} (lần {self.retry_count})...")
        
        self._cleanup()
        time.sleep(self.reconnect_delay)
        self._initialize_camera()
    
    def get_frame_buffer(self):
        """Lấy frame buffer (cho PIP)"""
        return self.frame_buffer.copy() if self.frame_buffer is not None else None
    
    def get_info(self):
        """Lấy thông tin camera"""
        return {
            'id': self.camera_id,
            'connected': self.connected,
            'type': self.camera_info['type'].value if self.camera_info['type'] else None,
            'resolution': self.camera_info['resolution'],
            'fps': self.camera_info['fps'],
            'frame_count': self.frame_count,
            'drop_count': self.drop_count,
            'drop_rate': self.drop_count / max(self.frame_count, 1),
            'retry_count': self.retry_count,
        }
    
    def set_property(self, prop_id, value):
        """Đặt thông số camera"""
        if self.cap is not None and self.connected:
            try:
                self.cap.set(prop_id, value)
                return True
            except Exception as e:
                print(f"❌ Không thể đặt property: {e}")
        return False
    
    def adjust_for_lighting(self, light_level):
        """Điều chỉnh camera theo điều kiện ánh sáng"""
        if not self.connected:
            return
        
        if light_level < 30:  # Tối
            self.set_property(cv2.CAP_PROP_BRIGHTNESS, 0.3)
            self.set_property(cv2.CAP_PROP_GAIN, 0.8)
        elif light_level > 200:  # Sáng
            self.set_property(cv2.CAP_PROP_BRIGHTNESS, 0.1)
            self.set_property(cv2.CAP_PROP_GAIN, 0.2)
        else:  # Bình thường
            self.set_property(cv2.CAP_PROP_BRIGHTNESS, 0.2)
            self.set_property(cv2.CAP_PROP_GAIN, 0.5)
    
    def is_opened(self):
        """Kiểm tra camera có đang mở không"""
        return self.connected and self.cap is not None and self.cap.isOpened()
    
    def _cleanup(self):
        """Dọn dẹp resource"""
        if self.cap is not None:
            try:
                self.cap.release()
            except:
                pass
            self.cap = None
        self.connected = False
    
    def release(self):
        """Giải phóng camera"""
        self._cleanup()
        print(f"📹 Camera {self.camera_id} đã được giải phóng")


class MultiCameraSystem:
    """Quản lý hệ thống đa camera"""
    def __init__(self, configs):
        self.cameras = {}
        self.active_camera = 'primary'
        
        # Khởi tạo các camera
        for cam_id, config in configs.items():
            if config is not None:
                self.cameras[cam_id] = Camera(config, camera_id=cam_id)
    
    def switch_camera(self, camera_id):
        """Chuyển đổi camera chính"""
        if camera_id in self.cameras and self.cameras[camera_id].connected:
            self.active_camera = camera_id
            return True
        return False
    
    def get_active_frame(self):
        """Lấy frame từ camera đang active"""
        if self.active_camera in self.cameras:
            return self.cameras[self.active_camera].read()
        return None, False
    
    def get_pip_frame(self, camera_id, size=(320, 240)):
        """Lấy frame cho PIP từ camera phụ"""
        if camera_id in self.cameras and camera_id != self.active_camera:
            frame, status = self.cameras[camera_id].read()
            if frame is not None:
                return cv2.resize(frame, size)
        return None
    
    def get_all_info(self):
        """Lấy thông tin tất cả camera"""
        return {cam_id: cam.get_info() for cam_id, cam in self.cameras.items()}
    
    def release_all(self):
        """Giải phóng tất cả camera"""
        for camera in self.cameras.values():
            camera.release()