# core/dashboard.py - Phiên bản đã sửa lỗi import
import cv2
import numpy as np
import datetime
from enum import Enum
import math

# Import với try-except để tránh lỗi
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠️ psutil not available, using simulated system stats")

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("⚠️ GPUtil not available, GPU stats will be simulated")

class DisplayMode(Enum):
    NORMAL = "normal"
    MINIMAL = "minimal"
    EXPERT = "expert"
    PIP = "pip"

class Dashboard:
    def __init__(self, config):
        self.config = config
        self.display_mode = DisplayMode.NORMAL
        self.show_help = False
        self.show_radar = True
        self.show_heatmap = False
        self.show_telemetry = True
        
        # Màu sắc
        self.COLOR_SAFE = config.COLOR_SAFE if hasattr(config, 'COLOR_SAFE') else (0, 255, 0)
        self.COLOR_WARNING = config.COLOR_WARNING if hasattr(config, 'COLOR_WARNING') else (0, 255, 255)
        self.COLOR_DANGER = config.COLOR_DANGER if hasattr(config, 'COLOR_DANGER') else (0, 0, 255)
        self.COLOR_INFO = config.COLOR_INFO if hasattr(config, 'COLOR_INFO') else (255, 255, 255)
        self.COLOR_SYSTEM = (100, 200, 255)  # Xanh nhạt
        
        # Dữ liệu hiển thị
        self.system_stats = {
            'cpu_usage': 0,
            'memory_usage': 0,
            'gpu_usage': 0,
            'temperature': 0,
            'disk_usage': 0,
        }
        
        # Thông tin cảnh báo
        self.alerts = []
        self.alert_history = []
        self.max_alerts = 5
        
        # Thông tin phương tiện
        self.vehicle_data = {
            'speed': 0,
            'location': (0, 0),
            'heading': 0,
            'rpm': 0,
            'fuel': 100,
        }
        
        # Biểu đồ
        self.fps_history = []
        self.speed_history = []
        self.obstacle_history = []
        self.max_history_points = 100
        
        # Zones và obstacles
        self.safety_zones = []
        self.warning_zones = []
        self.danger_zones = []
        self.obstacles = []
        
        # Heatmap data
        self.heatmap_data = np.zeros((100, 100), dtype=np.float32)
        self.heatmap_alpha = 0.1
        
        # Font và style
        self.font_small = cv2.FONT_HERSHEY_SIMPLEX
        self.font_medium = cv2.FONT_HERSHEY_DUPLEX
        self.font_large = cv2.FONT_HERSHEY_COMPLEX_SMALL
        
        # Timing
        self.last_update = datetime.datetime.now()
        
        # Load icons (nếu có)
        self.icons = self._load_icons()
    
    def _load_icons(self):
        """Tạo icons đơn giản bằng code"""
        icons = {}
        
        # Warning icon
        icon_size = 20
        warning_icon = np.zeros((icon_size, icon_size, 3), dtype=np.uint8)
        cv2.putText(warning_icon, "!", (7, 17), self.font_small, 0.7, (0, 255, 255), 2)
        icons['warning'] = warning_icon
        
        # Danger icon
        danger_icon = np.zeros((icon_size, icon_size, 3), dtype=np.uint8)
        cv2.putText(danger_icon, "X", (5, 17), self.font_small, 0.7, (0, 0, 255), 2)
        icons['danger'] = danger_icon
        
        # Camera icon
        camera_icon = np.zeros((icon_size, icon_size, 3), dtype=np.uint8)
        cv2.circle(camera_icon, (10, 10), 8, (255, 255, 255), 1)
        cv2.circle(camera_icon, (10, 10), 2, (255, 255, 255), -1)
        icons['camera'] = camera_icon
        
        return icons
    
    def update_system_stats(self):
        """Cập nhật thống kê hệ thống"""
        try:
            # CPU usage
            if PSUTIL_AVAILABLE:
                self.system_stats['cpu_usage'] = psutil.cpu_percent()
            else:
                self.system_stats['cpu_usage'] = 25.0 + np.random.rand() * 30
            
            # Memory usage
            if PSUTIL_AVAILABLE:
                memory = psutil.virtual_memory()
                self.system_stats['memory_usage'] = memory.percent
            else:
                self.system_stats['memory_usage'] = 40.0 + np.random.rand() * 30
            
            # Disk usage
            if PSUTIL_AVAILABLE:
                disk = psutil.disk_usage('/')
                self.system_stats['disk_usage'] = disk.percent
            else:
                self.system_stats['disk_usage'] = 30.0 + np.random.rand() * 20
            
            # GPU usage (nếu có)
            if GPU_AVAILABLE:
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        self.system_stats['gpu_usage'] = gpus[0].load * 100
                        self.system_stats['temperature'] = gpus[0].temperature
                    else:
                        self.system_stats['gpu_usage'] = 0
                        self.system_stats['temperature'] = 40 + np.random.rand() * 10
                except:
                    self.system_stats['gpu_usage'] = 0
                    self.system_stats['temperature'] = 40 + np.random.rand() * 10
            else:
                # Simulated GPU stats
                self.system_stats['gpu_usage'] = 15 + np.random.rand() * 20
                self.system_stats['temperature'] = 40 + np.random.rand() * 10
            
        except Exception as e:
            print(f"⚠️ Không thể cập nhật system stats: {e}")
            # Default values
            self.system_stats['cpu_usage'] = 25.0
            self.system_stats['memory_usage'] = 50.0
            self.system_stats['gpu_usage'] = 20.0
            self.system_stats['temperature'] = 45.0
            self.system_stats['disk_usage'] = 35.0
    
    def update_vehicle_data(self, speed=None, location=None, heading=None):
        """Cập nhật thông tin phương tiện"""
        if speed is not None:
            self.vehicle_data['speed'] = speed
            self.speed_history.append(speed)
            if len(self.speed_history) > self.max_history_points:
                self.speed_history.pop(0)
        
        if location is not None:
            self.vehicle_data['location'] = location
        
        if heading is not None:
            self.vehicle_data['heading'] = heading
    
    def add_alert(self, level, message, position=None):
        """Thêm cảnh báo mới"""
        timestamp = datetime.datetime.now()
        alert = {
            'level': level,
            'message': message,
            'timestamp': timestamp,
            'position': position,
            'duration': 5.0  # giây
        }
        
        self.alerts.insert(0, alert)
        self.alert_history.append(alert)
        
        # Giới hạn số lượng cảnh báo hiển thị
        if len(self.alerts) > self.max_alerts:
            self.alerts.pop()
    
    def update_obstacles(self, obstacles):
        """Cập nhật danh sách chướng ngại vật"""
        self.obstacles = obstacles
        
        # Cập nhật obstacle history
        self.obstacle_history.append(len(obstacles))
        if len(self.obstacle_history) > self.max_history_points:
            self.obstacle_history.pop(0)
        
        # Cập nhật heatmap
        self._update_heatmap(obstacles)
    
    def _update_heatmap(self, obstacles):
        """Cập nhật heatmap vị trí nguy hiểm"""
        if not obstacles:
            return
        
        # Giảm dần heatmap cũ
        self.heatmap_data *= (1 - self.heatmap_alpha)
        
        # Thêm heat mới
        for obstacle in obstacles:
            if 'position' in obstacle:
                x_norm = int(obstacle['position'][0] * 100)
                y_norm = int(obstacle['position'][1] * 100)
                x_norm = np.clip(x_norm, 0, 99)
                y_norm = np.clip(y_norm, 0, 99)
                
                # Tăng heat tại vị trí obstacle
                self.heatmap_data[y_norm, x_norm] += 0.5 * self.heatmap_alpha
                
                # Lan tỏa heat xung quanh
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        nx = np.clip(x_norm + dx, 0, 99)
                        ny = np.clip(y_norm + dy, 0, 99)
                        self.heatmap_data[ny, nx] += 0.1 * self.heatmap_alpha
    
    def update_zones(self, safety=None, warning=None, danger=None):
        """Cập nhật các vùng an toàn/cảnh báo/nguy hiểm"""
        if safety is not None:
            self.safety_zones = safety
        if warning is not None:
            self.warning_zones = warning
        if danger is not None:
            self.danger_zones = danger
    
    def update_fps(self, fps):
        """Cập nhật FPS"""
        self.fps_history.append(fps)
        if len(self.fps_history) > self.max_history_points:
            self.fps_history.pop(0)
    
    def draw(self, frame, fps, recording, camera_info=None):
        """Vẽ toàn bộ dashboard lên frame"""
        h, w = frame.shape[:2]
        
        # Cập nhật thời gian
        current_time = datetime.datetime.now()
        timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Tạo overlay để giảm ảnh hưởng đến hình ảnh gốc
        overlay = frame.copy()
        
        # Vẽ các thành phần theo chế độ hiển thị
        if self.display_mode == DisplayMode.MINIMAL:
            self._draw_minimal(overlay, fps, recording, timestamp, w, h)
        elif self.display_mode == DisplayMode.EXPERT:
            self._draw_expert(overlay, fps, recording, timestamp, w, h, camera_info)
        else:  # NORMAL
            self._draw_normal(overlay, fps, recording, timestamp, w, h, camera_info)
        
        # Thêm các thành phần chung
        self._draw_alerts(overlay, w, h)
        
        # Blend overlay với frame gốc
        alpha = 0.7 if self.display_mode != DisplayMode.MINIMAL else 0.8
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Vẽ các thành phần không cần blend
        self._draw_zones(frame)
        self._draw_obstacles(frame)
        
        # Vẽ radar nếu bật
        if self.show_radar:
            self._draw_radar(frame)
        
        # Vẽ heatmap nếu bật
        if self.show_heatmap:
            self._draw_heatmap(frame)
        
        # Vẽ telemetry nếu bật
        if self.show_telemetry:
            self._draw_telemetry(frame)
        
        # Vẽ help nếu bật
        if self.show_help:
            self._draw_help(frame, w, h)
        
        return frame
    
    def _draw_normal(self, overlay, fps, recording, timestamp, w, h, camera_info):
        """Vẽ dashboard chế độ normal"""
        # Header background
        cv2.rectangle(overlay, (0, 0), (w, 40), (0, 0, 0), -1)
        cv2.rectangle(overlay, (0, h-100), (w, h), (0, 0, 0), -1)
        
        # Title
        cv2.putText(overlay, "🚗 Obstacle Detection System", (10, 30),
                   cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
        
        # Status indicators
        status_x = w - 200
        self._draw_status_indicator(overlay, status_x, 20, recording, "REC")
        
        # Main info panel (trái)
        y_start = 60
        lines_left = [
            f"FPS: {int(fps)}",
            f"Time: {timestamp}",
            f"Speed: {self.vehicle_data['speed']:.1f} km/h",
            f"Mode: {self.display_mode.value.upper()}",
        ]
        
        if camera_info:
            lines_left.append(f"Camera: {camera_info.get('type', 'Unknown')}")
            lines_left.append(f"Res: {camera_info.get('resolution', (0,0))[0]}x{camera_info.get('resolution', (0,0))[1]}")
        
        for i, line in enumerate(lines_left):
            color = self.COLOR_INFO if i > 0 else (0, 255, 255)
            cv2.putText(overlay, line, (20, y_start + i * 30),
                       self.font_small, 0.7, color, 2)
        
        # System info panel (phải)
        self.update_system_stats()
        lines_right = [
            f"CPU: {self.system_stats['cpu_usage']:.1f}%",
            f"RAM: {self.system_stats['memory_usage']:.1f}%",
            f"GPU: {self.system_stats['gpu_usage']:.1f}%",
            f"Temp: {self.system_stats['temperature']:.1f}°C",
            f"Obstacles: {len(self.obstacles)}",
        ]
        
        for i, line in enumerate(lines_right):
            x_pos = w - 220
            # Màu sắc theo mức độ
            if 'CPU' in line or 'GPU' in line:
                usage = float(line.split(':')[1].replace('%', '').strip())
                color = self.COLOR_SAFE if usage < 70 else self.COLOR_WARNING if usage < 90 else self.COLOR_DANGER
            elif 'Temp' in line:
                temp = float(line.split(':')[1].replace('°C', '').strip())
                color = self.COLOR_SAFE if temp < 70 else self.COLOR_WARNING if temp < 85 else self.COLOR_DANGER
            else:
                color = self.COLOR_INFO
            
            cv2.putText(overlay, line, (x_pos, y_start + i * 30),
                       self.font_small, 0.7, color, 2)
        
        # Vẽ biểu đồ FPS nhỏ
        self._draw_mini_graph(overlay, w - 200, h - 90, 180, 80, 
                            self.fps_history, "FPS", (0, 255, 255))
    
    def _draw_minimal(self, overlay, fps, recording, timestamp, w, h):
        """Vẽ dashboard chế độ minimal"""
        # Chỉ hiển thị thông tin cần thiết
        lines = [
            f"{timestamp}",
            f"FPS: {int(fps)} | SPD: {self.vehicle_data['speed']:.0f}km/h",
            f"OBS: {len(self.obstacles)} | REC: {'●' if recording else '○'}"
        ]
        
        # Nền bán trong suốt
        cv2.rectangle(overlay, (0, 0), (w, 90), (0, 0, 0), -1)
        
        for i, line in enumerate(lines):
            y_pos = 30 + i * 25
            color = (200, 200, 200) if i > 0 else (255, 255, 255)
            font_size = 0.6 if i > 0 else 0.7
            cv2.putText(overlay, line, (20, y_pos),
                       self.font_small, font_size, color, 1)
    
    def _draw_expert(self, overlay, fps, recording, timestamp, w, h, camera_info):
        """Vẽ dashboard chế độ expert"""
        # Nền thông tin đầy đủ
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        alpha = 0.3
        overlay = cv2.addWeighted(overlay, alpha, np.zeros_like(overlay), 1-alpha, 0)
        
        # Multiple panels
        panel_width = w // 3 - 10
        
        # Panel 1: System Info
        self._draw_panel(overlay, 10, 10, panel_width, h-20, "SYSTEM INFO")
        sys_info = [
            f"FPS: {int(fps)} (Avg: {np.mean(self.fps_history[-10:]) if len(self.fps_history) >= 10 else fps:.1f})",
            f"Frame: {self.fps_history[-1] if self.fps_history else 0}",
            f"CPU: {self.system_stats['cpu_usage']:.1f}%",
            f"RAM: {self.system_stats['memory_usage']:.1f}%",
            f"GPU: {self.system_stats['gpu_usage']:.1f}%",
            f"Temp: {self.system_stats['temperature']:.1f}°C",
            f"Disk: {self.system_stats['disk_usage']:.1f}%",
        ]
        
        for i, info in enumerate(sys_info):
            y_pos = 60 + i * 30
            cv2.putText(overlay, info, (30, y_pos), 
                       self.font_small, 0.6, self.COLOR_INFO, 1)
        
        # Panel 2: Vehicle Info
        self._draw_panel(overlay, 20 + panel_width, 10, panel_width, h-20, "VEHICLE INFO")
        vehicle_info = [
            f"Speed: {self.vehicle_data['speed']:.1f} km/h",
            f"Heading: {self.vehicle_data['heading']:.0f}°",
            f"Location: {self.vehicle_data['location'][0]:.4f}, {self.vehicle_data['location'][1]:.4f}",
            f"RPM: {self.vehicle_data['rpm']:.0f}",
            f"Fuel: {self.vehicle_data['fuel']:.1f}%",
            f"Obstacles: {len(self.obstacles)}",
            f"Recording: {'ON' if recording else 'OFF'}",
        ]
        
        for i, info in enumerate(vehicle_info):
            y_pos = 60 + i * 30
            cv2.putText(overlay, info, (40 + panel_width, y_pos),
                       self.font_small, 0.6, self.COLOR_INFO, 1)
        
        # Panel 3: Camera Info
        if camera_info:
            self._draw_panel(overlay, 30 + panel_width*2, 10, panel_width, h-20, "CAMERA INFO")
            cam_info = [
                f"Type: {camera_info.get('type', 'Unknown')}",
                f"Res: {camera_info.get('resolution', (0,0))[0]}x{camera_info.get('resolution', (0,0))[1]}",
                f"FPS: {camera_info.get('fps', 0):.1f}",
                f"Connected: {'Yes' if camera_info.get('connected') else 'No'}",
                f"Frame Count: {camera_info.get('frame_count', 0)}",
                f"Drop Rate: {camera_info.get('drop_rate', 0)*100:.1f}%",
            ]
            
            for i, info in enumerate(cam_info):
                y_pos = 60 + i * 30
                cv2.putText(overlay, info, (50 + panel_width*2, y_pos),
                           self.font_small, 0.6, self.COLOR_INFO, 1)
    
    def _draw_panel(self, overlay, x, y, w, h, title):
        """Vẽ một panel thông tin"""
        # Background
        cv2.rectangle(overlay, (x, y), (x+w, y+h), (30, 30, 30), -1)
        cv2.rectangle(overlay, (x, y), (x+w, y+h), (100, 100, 100), 1)
        
        # Title
        cv2.putText(overlay, title, (x+10, y+25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    def _draw_status_indicator(self, overlay, x, y, status, label):
        """Vẽ indicator trạng thái"""
        color = (0, 0, 255) if status else (100, 100, 100)
        radius = 8
        
        # Đèn indicator
        cv2.circle(overlay, (x, y), radius, color, -1)
        if status:
            cv2.circle(overlay, (x, y), radius, (255, 255, 255), 1)
        
        # Label
        cv2.putText(overlay, label, (x + 15, y + 5),
                   self.font_small, 0.6, color, 2)
    
    def _draw_mini_graph(self, overlay, x, y, w, h, data, title, color):
        """Vẽ biểu đồ mini"""
        if len(data) < 2:
            return
        
        # Background
        cv2.rectangle(overlay, (x, y), (x+w, y+h), (20, 20, 20), -1)
        
        # Title
        cv2.putText(overlay, title, (x+5, y+15),
                   self.font_small, 0.5, (200, 200, 200), 1)
        
        # Vẽ biểu đồ
        points = []
        max_val = max(data) if data else 1
        min_val = min(data) if data else 0
        
        for i, value in enumerate(data[-w:]):  # Giới hạn số điểm bằng chiều rộng
            x_pos = x + i
            y_pos = int(y + h - (value - min_val) / max(1, max_val - min_val) * (h - 20))
            points.append((x_pos, y_pos))
        
        # Vẽ đường
        for i in range(1, len(points)):
            cv2.line(overlay, points[i-1], points[i], color, 1)
        
        # Vẽ giá trị hiện tại
        if data:
            current_val = data[-1]
            cv2.putText(overlay, f"{current_val:.0f}", (x+w-40, y+15),
                       self.font_small, 0.5, color, 1)
    
    def _draw_alerts(self, overlay, w, h):
        """Vẽ các cảnh báo"""
        if not self.alerts:
            return
        
        # Xóa cảnh báo hết hạn
        current_time = datetime.datetime.now()
        self.alerts = [alert for alert in self.alerts 
                      if (current_time - alert['timestamp']).seconds < alert['duration']]
        
        # Vẽ từng cảnh báo
        alert_x = 10
        alert_y = h - 150
        
        for i, alert in enumerate(self.alerts[:3]):  # Tối đa 3 cảnh báo
            # Màu sắc theo mức độ
            if alert['level'] == 'high':
                bg_color = (50, 50, 150)  # Đỏ đậm
                text_color = (0, 0, 255)
                icon = self.icons.get('danger')
            elif alert['level'] == 'medium':
                bg_color = (50, 100, 100)  # Vàng đậm
                text_color = (0, 255, 255)
                icon = self.icons.get('warning')
            else:
                bg_color = (50, 100, 50)   # Xanh đậm
                text_color = (0, 255, 0)
                icon = None
            
            # Nền cảnh báo
            alert_h = 40
            cv2.rectangle(overlay, (alert_x, alert_y), 
                         (alert_x + 400, alert_y + alert_h), bg_color, -1)
            cv2.rectangle(overlay, (alert_x, alert_y),
                         (alert_x + 400, alert_y + alert_h), text_color, 1)
            
            # Icon
            if icon is not None:
                overlay[alert_y+10:alert_y+30, alert_x+10:alert_x+30] = icon
            
            # Text
            time_str = alert['timestamp'].strftime("%H:%M:%S")
            message = f"[{time_str}] {alert['message']}"
            cv2.putText(overlay, message, (alert_x + 40, alert_y + 25),
                       self.font_small, 0.6, text_color, 1)
            
            alert_y -= 45  # Khoảng cách giữa các cảnh báo
    
    def _draw_zones(self, frame):
        """Vẽ các vùng an toàn/cảnh báo/nguy hiểm"""
        h, w = frame.shape[:2]
        
        # Vẽ vùng an toàn (xanh lá)
        for zone in self.safety_zones:
            points = np.array(zone, dtype=np.int32)
            points = points.reshape((-1, 1, 2))
            cv2.polylines(frame, [points], True, self.COLOR_SAFE, 1)
            cv2.fillPoly(frame, [points], (*self.COLOR_SAFE, 30))
        
        # Vẽ vùng cảnh báo (vàng)
        for zone in self.warning_zones:
            points = np.array(zone, dtype=np.int32)
            points = points.reshape((-1, 1, 2))
            cv2.polylines(frame, [points], True, self.COLOR_WARNING, 2)
            cv2.fillPoly(frame, [points], (*self.COLOR_WARNING, 20))
        
        # Vẽ vùng nguy hiểm (đỏ)
        for zone in self.danger_zones:
            points = np.array(zone, dtype=np.int32)
            points = points.reshape((-1, 1, 2))
            cv2.polylines(frame, [points], True, self.COLOR_DANGER, 3)
            cv2.fillPoly(frame, [points], (*self.COLOR_DANGER, 15))
    
    def _draw_obstacles(self, frame):
        """Vẽ chướng ngại vật"""
        for obstacle in self.obstacles:
            if 'bbox' in obstacle:
                x1, y1, x2, y2 = obstacle['bbox']
                confidence = obstacle.get('confidence', 0)
                
                # Màu sắc theo confidence
                if confidence > 0.7:
                    color = self.COLOR_DANGER
                    thickness = 3
                elif confidence > 0.4:
                    color = self.COLOR_WARNING
                    thickness = 2
                else:
                    color = self.COLOR_SAFE
                    thickness = 1
                
                # Vẽ bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                
                # Vẽ label
                label = f"{obstacle.get('class', 'Obj')}: {confidence:.1%}"
                label_size, baseline = cv2.getTextSize(label, self.font_small, 0.5, 1)
                cv2.rectangle(frame, (x1, y1-label_size[1]-5),
                             (x1+label_size[0], y1), color, -1)
                cv2.putText(frame, label, (x1, y1-5),
                           self.font_small, 0.5, (255, 255, 255), 1)
    
    def _draw_radar(self, frame):
        """Vẽ radar hiển thị góc và khoảng cách"""
        h, w = frame.shape[:2]
        radar_size = 150
        radar_x = w - radar_size - 20
        radar_y = 20
        
        # Nền radar
        cv2.circle(frame, (radar_x + radar_size//2, radar_y + radar_size//2),
                  radar_size//2, (30, 30, 30), -1)
        cv2.circle(frame, (radar_x + radar_size//2, radar_y + radar_size//2),
                  radar_size//2, (100, 100, 100), 1)
        
        # Vòng tròn đồng tâm
        for r in [radar_size//4, radar_size//2, radar_size*3//4]:
            cv2.circle(frame, (radar_x + radar_size//2, radar_y + radar_size//2),
                      r, (50, 50, 50), 1)
        
        # Trục
        center_x, center_y = radar_x + radar_size//2, radar_y + radar_size//2
        cv2.line(frame, (center_x, radar_y), (center_x, radar_y+radar_size),
                (100, 100, 100), 1)
        cv2.line(frame, (radar_x, center_y), (radar_x+radar_size, center_y),
                (100, 100, 100), 1)
        
        # Vẽ chướng ngại vật trên radar
        for obstacle in self.obstacles:
            if 'position' in obstacle:
                x_norm, y_norm = obstacle['position']
                # Chuyển đổi tọa độ chuẩn hóa sang tọa độ radar
                radar_pos_x = center_x + int((x_norm - 0.5) * radar_size)
                radar_pos_y = center_y + int((y_norm - 0.5) * radar_size)
                
                # Màu theo distance/confidence
                color = self.COLOR_DANGER if obstacle.get('distance', 0) < 5 else self.COLOR_WARNING
                cv2.circle(frame, (radar_pos_x, radar_pos_y), 4, color, -1)
        
        # Vẽ vehicle direction
        direction_length = radar_size//2 - 10
        angle_rad = math.radians(self.vehicle_data['heading'])
        end_x = center_x + int(direction_length * math.sin(angle_rad))
        end_y = center_y - int(direction_length * math.cos(angle_rad))
        
        cv2.arrowedLine(frame, (center_x, center_y), (end_x, end_y),
                       (0, 255, 0), 2)
        
        # Title
        cv2.putText(frame, "RADAR", (radar_x + 10, radar_y + 15),
                   self.font_small, 0.6, (200, 200, 200), 1)
    
    def _draw_heatmap(self, frame):
        """Vẽ heatmap khu vực nguy hiểm thường xuyên"""
        h, w = frame.shape[:2]
        heatmap_size = 120
        heatmap_x = w - heatmap_size - 20
        heatmap_y = 180
        
        # Resize heatmap data
        heatmap_display = cv2.resize(self.heatmap_data, (heatmap_size, heatmap_size))
        
        # Áp dụng colormap
        heatmap_colored = cv2.applyColorMap((heatmap_display * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # Blend với frame
        roi = frame[heatmap_y:heatmap_y+heatmap_size, heatmap_x:heatmap_x+heatmap_size]
        blended = cv2.addWeighted(roi, 0.5, heatmap_colored, 0.5, 0)
        frame[heatmap_y:heatmap_y+heatmap_size, heatmap_x:heatmap_x+heatmap_size] = blended
        
        # Border và title
        cv2.rectangle(frame, (heatmap_x, heatmap_y),
                     (heatmap_x+heatmap_size, heatmap_y+heatmap_size),
                     (255, 255, 255), 1)
        
        cv2.putText(frame, "HEATMAP", (heatmap_x + 10, heatmap_y + 15),
                   self.font_small, 0.6, (255, 255, 255), 1)
        
        # Legend
        legend_y = heatmap_y + heatmap_size + 20
        cv2.putText(frame, "Low", (heatmap_x, legend_y),
                   self.font_small, 0.5, (100, 100, 255), 1)
        cv2.putText(frame, "High", (heatmap_x + heatmap_size - 40, legend_y),
                   self.font_small, 0.5, (255, 100, 100), 1)
    
    def _draw_telemetry(self, frame):
        """Vẽ thông số phương tiện chi tiết"""
        h, w = frame.shape[:2]
        
        # Tạo một overlay nhỏ ở góc dưới trái
        telemetry_x = 20
        telemetry_y = h - 200
        
        # Nền
        cv2.rectangle(frame, (telemetry_x, telemetry_y),
                     (telemetry_x + 180, telemetry_y + 100),
                     (0, 0, 0, 128), -1)
        
        # Speed gauge đơn giản
        speed = self.vehicle_data['speed']
        max_speed = 120  # km/h
        
        # Vẽ gauge
        gauge_center = (telemetry_x + 40, telemetry_y + 40)
        gauge_radius = 35
        
        # Background circle
        cv2.circle(frame, gauge_center, gauge_radius, (50, 50, 50), -1)
        
        # Speed arc
        angle = min(speed / max_speed * 270, 270)  # 0-270 độ
        angle_rad = math.radians(angle - 135)  # Offset để bắt đầu từ bottom-left
        
        end_x = gauge_center[0] + int(gauge_radius * math.cos(angle_rad))
        end_y = gauge_center[1] + int(gauge_radius * math.sin(angle_rad))
        
        # Màu theo tốc độ
        if speed < 60:
            color = self.COLOR_SAFE
        elif speed < 90:
            color = self.COLOR_WARNING
        else:
            color = self.COLOR_DANGER
        
        # Vẽ kim
        cv2.line(frame, gauge_center, (end_x, end_y), color, 3)
        
        # Hiển thị tốc độ
        cv2.putText(frame, f"{speed:.0f}", (gauge_center[0]-15, gauge_center[1]+5),
                   cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2)
        cv2.putText(frame, "km/h", (gauge_center[0]-20, gauge_center[1]+25),
                   self.font_small, 0.5, (200, 200, 200), 1)
        
        # Thông tin bổ sung
        info_x = telemetry_x + 90
        info_lines = [
            f"RPM: {self.vehicle_data['rpm']:.0f}",
            f"Fuel: {self.vehicle_data['fuel']:.0f}%",
            f"Heading: {self.vehicle_data['heading']:.0f}°",
        ]
        
        for i, line in enumerate(info_lines):
            cv2.putText(frame, line, (info_x, telemetry_y + 30 + i*20),
                       self.font_small, 0.5, self.COLOR_INFO, 1)
    
    def _draw_help(self, frame, w, h):
        """Vẽ hướng dẫn phím tắt"""
        # Nền help
        help_x = w // 2 - 200
        help_y = h // 2 - 150
        
        cv2.rectangle(frame, (help_x, help_y), (help_x + 400, help_y + 300),
                     (0, 0, 0, 200), -1)
        cv2.rectangle(frame, (help_x, help_y), (help_x + 400, help_y + 300),
                     (255, 255, 255), 2)
        
        # Title
        cv2.putText(frame, "HELP - KEYBOARD SHORTCUTS", 
                   (help_x + 50, help_y + 30),
                   cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
        
        # Hotkeys
        hotkeys = [
            ("Q", "Quit application"),
            ("S", "Take snapshot"),
            ("R", "Toggle recording"),
            ("P", "Toggle PIP mode"),
            ("D", "Cycle display modes"),
            ("E", "Toggle enhancement"),
            ("H", "Show/Hide this help"),
            ("1/2/3", "Adjust brightness"),
            ("F", "Toggle fullscreen"),
            ("C", "Calibrate camera"),
        ]
        
        for i, (key, desc) in enumerate(hotkeys):
            y_pos = help_y + 70 + i * 25
            cv2.putText(frame, f"{key}:", (help_x + 30, y_pos),
                       self.font_small, 0.6, (0, 255, 255), 1)
            cv2.putText(frame, desc, (help_x + 100, y_pos),
                       self.font_small, 0.6, (200, 200, 200), 1)
        
        # Footer
        cv2.putText(frame, "Press H to close", 
                   (help_x + 130, help_y + 280),
                   self.font_small, 0.6, (150, 150, 150), 1)
    
    def cycle_display_mode(self):
        """Chuyển đổi giữa các chế độ hiển thị"""
        modes = list(DisplayMode)
        current_index = modes.index(self.display_mode)
        next_index = (current_index + 1) % len(modes)
        self.display_mode = modes[next_index]
        
        print(f"🔄 Display mode changed to: {self.display_mode.value}")
        return self.display_mode
    
    def toggle_radar(self):
        """Bật/tắt radar"""
        self.show_radar = not self.show_radar
        print(f"🔄 Radar {'enabled' if self.show_radar else 'disabled'}")
    
    def toggle_heatmap(self):
        """Bật/tắt heatmap"""
        self.show_heatmap = not self.show_heatmap
        print(f"🔄 Heatmap {'enabled' if self.show_heatmap else 'disabled'}")
    
    def toggle_telemetry(self):
        """Bật/tắt telemetry"""
        self.show_telemetry = not self.show_telemetry
        print(f"🔄 Telemetry {'enabled' if self.show_telemetry else 'disabled'}")
    
    def toggle_help(self):
        """Bật/tắt help"""
        self.show_help = not self.show_help
        print(f"🔄 Help {'shown' if self.show_help else 'hidden'}")