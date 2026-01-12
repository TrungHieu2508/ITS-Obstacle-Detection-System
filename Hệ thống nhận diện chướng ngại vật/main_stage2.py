# main_fixed.py - Phiên bản chính đã sửa lỗi
import cv2
import sys
import time
import traceback
import numpy as np
from datetime import datetime

# Import config đã sửa
import config_fixed as config

# Import core modules
from core.camera import MultiCameraSystem
from core.enhancer import ImageEnhancer
from core.fps import FPSCounter
from core.dashboard import Dashboard, DisplayMode
from core.recorder import VideoRecorder
from core.sensors import VehicleSensors
from core.alert_system import AlertSystem, AlertLevel, AlertType
from utils.logger import StructuredLogger
from utils.performance import PerformanceMonitor

class ObstacleDetectionSystem:
    def __init__(self, config):
        self.config = config
        self.running = False
        
        print("=" * 60)
        print("🚗 OBSTACLE DETECTION SYSTEM - STAGE 2 (FIXED)")
        print("=" * 60)
        
        try:
            # Khởi tạo các component
            self.logger = StructuredLogger(config)
            print("✅ Logger initialized")
            
            self.performance_monitor = PerformanceMonitor(config)
            print("✅ Performance monitor initialized")
            
            self.alert_system = AlertSystem(config)
            print("✅ Alert system initialized")
            
            # Camera system
            self.camera_system = MultiCameraSystem(config.CAMERA_SOURCES)
            print("✅ Camera system initialized")
            self.active_camera = 'primary'
            
            # Image processing
            self.enhancer = ImageEnhancer(config)
            print("✅ Image enhancer initialized")
            self.fps_counter = FPSCounter(config.FPS_BUFFER_SIZE)
            
            # Display and UI
            self.dashboard = Dashboard(config)
            print("✅ Dashboard initialized")
            self.fullscreen = False
            self.window_name = "🚗 Obstacle Detection System v2.0"
            
            # Recording
            camera_info = {}
            if self.active_camera in self.camera_system.cameras:
                camera = self.camera_system.cameras[self.active_camera]
                camera_info = camera.get_info() if hasattr(camera, 'get_info') else {}
            
            self.recorder = VideoRecorder(config, camera_info)
            print("✅ Video recorder initialized")
            
            # Sensors
            self.sensors = VehicleSensors(config)
            print("✅ Vehicle sensors initialized")
            
            # Obstacle detection (placeholder)
            self.obstacle_detector = self._create_obstacle_detector()
            print("✅ Obstacle detector initialized")
            
            # Register callbacks
            self._register_callbacks()
            print("✅ Callbacks registered")
            
            # Statistics
            self.frame_count = 0
            self.start_time = time.time()
            self.last_snapshot_time = 0
            
            # PIP settings
            self.show_pip = config.SHOW_PIP if hasattr(config, 'SHOW_PIP') else False
            
            print("=" * 60)
            print(f"📊 Resolution: {config.FRAME_WIDTH}x{config.FRAME_HEIGHT}")
            print(f"🎥 Active camera: {self.active_camera}")
            print(f"💾 Recording: {'ON' if config.RECORD_VIDEO else 'OFF'}")
            print("=" * 60)
            
        except Exception as e:
            print(f"❌ Error initializing system: {e}")
            traceback.print_exc()
            raise
    
    def _create_obstacle_detector(self):
        """Tạo obstacle detector (placeholder cho stage sau)"""
        # Ở stage 2, chỉ mô phỏng detection
        class MockDetector:
            def detect(self, frame):
                # Mock detection - sẽ thay bằng YOLO/SSD ở stage sau
                h, w = frame.shape[:2]
                
                # Tạo mock obstacles ngẫu nhiên
                obstacles = []
                if np.random.random() < 0.3:  # 30% chance
                    for _ in range(np.random.randint(0, 3)):
                        x1 = np.random.randint(0, w - 100)
                        y1 = np.random.randint(0, h - 100)
                        x2 = x1 + np.random.randint(50, 200)
                        y2 = y1 + np.random.randint(50, 200)
                        
                        confidence = np.random.uniform(0.3, 0.95)
                        
                        obstacles.append({
                            'bbox': (x1, y1, x2, y2),
                            'confidence': confidence,
                            'class': 'obstacle',
                            'distance': np.random.uniform(1, 20),
                            'position': (x1 / w, y1 / h)
                        })
                
                return obstacles
        
        return MockDetector()
    
    def _register_callbacks(self):
        """Đăng ký các callback"""
        try:
            # Performance alert callback
            self.performance_monitor.register_alert_callback(
                lambda alert_type, msg, data: self.alert_system.trigger_alert(
                    AlertLevel.MEDIUM if 'low' in alert_type else AlertLevel.LOW,
                    f"Performance: {msg}",
                    AlertType.VISUAL,
                    data=data
                )
            )
            
            # Performance adjustment callback
            self.performance_monitor.register_adjustment_callback(
                self._handle_performance_adjustment
            )
            
            # Sensor data callback
            self.sensors.register_callback(self._handle_sensor_data)
            
        except Exception as e:
            print(f"⚠️ Error registering callbacks: {e}")
    
    def _handle_performance_adjustment(self, settings):
        """Xử lý điều chỉnh performance"""
        try:
            # Có thể điều chỉnh enhancer dựa trên settings
            if settings['enhancement_level'] == 'low':
                self.enhancer.use_gamma = False
                self.enhancer.use_hist = False
            elif settings['enhancement_level'] == 'medium':
                self.enhancer.use_gamma = True
                self.enhancer.use_hist = False
            else:  # high
                self.enhancer.use_gamma = True
                self.enhancer.use_hist = True
        except Exception as e:
            print(f"⚠️ Error handling performance adjustment: {e}")
    
    def _handle_sensor_data(self, sensor_data):
        """Xử lý dữ liệu cảm biến"""
        try:
            # Cập nhật dashboard
            gps_data = sensor_data['gps']
            vehicle_data = sensor_data['vehicle']
            
            self.dashboard.update_vehicle_data(
                speed=gps_data['speed'],
                location=(gps_data['latitude'], gps_data['longitude']),
                heading=gps_data['heading']
            )
            
            # Cập nhật RPM và fuel
            self.dashboard.vehicle_data['rpm'] = vehicle_data['rpm']
            self.dashboard.vehicle_data['fuel'] = vehicle_data['fuel_level']
            
            # Kiểm tra cảnh báo dựa trên sensor data
            self._check_sensor_alerts(sensor_data)
        except Exception as e:
            print(f"⚠️ Error handling sensor data: {e}")
    
    def _check_sensor_alerts(self, sensor_data):
        """Kiểm tra cảnh báo từ cảm biến"""
        try:
            vehicle_data = sensor_data['vehicle']
            gps_data = sensor_data['gps']
            
            # Kiểm tra tốc độ
            if gps_data['speed'] > 100:  # >100 km/h
                self.alert_system.trigger_alert(
                    AlertLevel.MEDIUM,
                    f"High speed: {gps_data['speed']:.0f} km/h",
                    AlertType.VISUAL
                )
            
            # Kiểm tra nhiệt độ động cơ
            if vehicle_data['engine_temp'] > 100:
                self.alert_system.trigger_alert(
                    AlertLevel.HIGH,
                    f"Engine overheating: {vehicle_data['engine_temp']:.1f}°C",
                    AlertType.VISUAL
                )
            
            # Kiểm tra nhiên liệu
            if vehicle_data['fuel_level'] < 15:
                self.alert_system.trigger_alert(
                    AlertLevel.LOW,
                    f"Low fuel: {vehicle_data['fuel_level']:.1f}%",
                    AlertType.VISUAL
                )
        except Exception as e:
            print(f"⚠️ Error checking sensor alerts: {e}")
    
    def _process_frame(self, frame):
        """Xử lý một frame"""
        if frame is None:
            return None, 0, []
        
        start_process_time = time.time() * 1000
        
        try:
            # 1. Enhancement
            enhanced_frame = self.enhancer.process(frame)
            
            # 2. Obstacle detection
            obstacles = self.obstacle_detector.detect(enhanced_frame)
            
            # 3. Cập nhật dashboard với obstacles
            self.dashboard.update_obstacles(obstacles)
            
            # 4. Kiểm tra cảnh báo obstacles
            self._check_obstacle_alerts(obstacles)
            
            # 5. Tính FPS
            fps_avg = self.fps_counter.update()
            self.dashboard.update_fps(fps_avg)
            
            # 6. Cập nhật performance metrics
            process_time = time.time() * 1000 - start_process_time
            self.performance_monitor.update_frame_metrics(fps_avg, process_time)
            
            return enhanced_frame, fps_avg, obstacles
            
        except Exception as e:
            print(f"❌ Error processing frame: {e}")
            return frame, 0, []
    
    def _check_obstacle_alerts(self, obstacles):
        """Kiểm tra cảnh báo từ chướng ngại vật"""
        if not obstacles:
            return
        
        try:
            # Phân loại obstacles theo confidence và distance
            high_conf_obstacles = [o for o in obstacles if o['confidence'] > 0.7]
            close_obstacles = [o for o in obstacles if o.get('distance', 100) < 5]
            
            if close_obstacles:
                # Obstacle gần -> cảnh báo cao
                self.alert_system.trigger_alert(
                    AlertLevel.HIGH,
                    f"Close obstacle detected! Distance: {close_obstacles[0]['distance']:.1f}m",
                    AlertType.VISUAL,
                    position=close_obstacles[0].get('position')
                )
                
                # Trigger event recording
                if hasattr(self.config, 'RECORD_EVENTS_ONLY') and self.config.RECORD_EVENTS_ONLY:
                    self.recorder.trigger_event('obstacle_detected', {
                        'count': len(close_obstacles),
                        'distances': [o['distance'] for o in close_obstacles]
                    })
            
            elif high_conf_obstacles:
                # Obstacle confidence cao -> cảnh báo trung bình
                self.alert_system.trigger_alert(
                    AlertLevel.MEDIUM,
                    f"High confidence obstacle detected",
                    AlertType.VISUAL,
                    position=high_conf_obstacles[0].get('position')
                )
        except Exception as e:
            print(f"⚠️ Error checking obstacle alerts: {e}")
    
    def _handle_keypress(self, key):
        """Xử lý phím bấm"""
        # Quit
        if key == self.config.HOTKEYS['quit']:
            self.running = False
            print("👋 Shutting down...")
        
        # Snapshot
        elif key == self.config.HOTKEYS['snapshot']:
            current_time = time.time()
            if current_time - self.last_snapshot_time > 1.0:  # Debounce
                self._take_snapshot()
                self.last_snapshot_time = current_time
        
        # Toggle recording
        elif key == self.config.HOTKEYS['toggle_record']:
            if self.recorder.recording:
                self.recorder.stop()
                self.alert_system.trigger_alert(
                    AlertLevel.INFO,
                    "Recording stopped",
                    AlertType.VISUAL
                )
            else:
                if self.recorder.start():
                    self.alert_system.trigger_alert(
                        AlertLevel.INFO,
                        "Recording started",
                        AlertType.VISUAL
                    )
        
        # Toggle PIP
        elif key == self.config.HOTKEYS['toggle_pip']:
            self.show_pip = not self.show_pip
            status = "ON" if self.show_pip else "OFF"
            self.alert_system.trigger_alert(
                AlertLevel.INFO,
                f"PIP {status}",
                AlertType.VISUAL
            )
        
        # Cycle display mode
        elif key == self.config.HOTKEYS['cycle_display']:
            new_mode = self.dashboard.cycle_display_mode()
            self.alert_system.trigger_alert(
                AlertLevel.INFO,
                f"Display mode: {new_mode.value}",
                AlertType.VISUAL
            )
        
        # Toggle enhancement
        elif key == self.config.HOTKEYS['toggle_enhancement']:
            gamma_state = self.enhancer.toggle_gamma()
            status = "ON" if gamma_state else "OFF"
            self.alert_system.trigger_alert(
                AlertLevel.INFO,
                f"Gamma correction {status}",
                AlertType.VISUAL
            )
        
        # Calibrate
        elif key == self.config.HOTKEYS['calibrate']:
            self._calibrate_system()
        
        # Show help
        elif key == self.config.HOTKEYS['show_help']:
            self.dashboard.toggle_help()
        
        # Adjust brightness
        elif key == ord('1'):  # Decrease brightness
            self.enhancer.adjust_gamma(-0.1)
            self.alert_system.trigger_alert(
                AlertLevel.INFO,
                f"Gamma: {self.enhancer.gamma:.1f}",
                AlertType.VISUAL
            )
        elif key == ord('2'):  # Reset brightness
            self.enhancer.gamma = self.config.GAMMA_VALUE
            self.alert_system.trigger_alert(
                AlertLevel.INFO,
                f"Gamma reset: {self.enhancer.gamma:.1f}",
                AlertType.VISUAL
            )
        elif key == ord('3'):  # Increase brightness
            self.enhancer.adjust_gamma(0.1)
            self.alert_system.trigger_alert(
                AlertLevel.INFO,
                f"Gamma: {self.enhancer.gamma:.1f}",
                AlertType.VISUAL
            )
        
        # Toggle fullscreen
        elif key == ord('f'):
            self.fullscreen = not self.fullscreen
            if self.fullscreen:
                cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        
        # Test alert
        elif key == ord('t'):
            self.alert_system.test_alert(AlertLevel.HIGH)
        
        # Switch camera
        elif key == ord('c'):
            cameras = list(self.camera_system.cameras.keys())
            if cameras:
                current_index = cameras.index(self.active_camera) if self.active_camera in cameras else 0
                next_index = (current_index + 1) % len(cameras)
                if self.camera_system.switch_camera(cameras[next_index]):
                    self.active_camera = cameras[next_index]
                    self.alert_system.trigger_alert(
                        AlertLevel.INFO,
                        f"Switched to {self.active_camera} camera",
                        AlertType.VISUAL
                    )
    
    def _take_snapshot(self):
        """Chụp ảnh màn hình"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"snapshot_{timestamp}.jpg"
            
            # Lấy camera active
            if self.active_camera in self.camera_system.cameras:
                camera = self.camera_system.cameras[self.active_camera]
                if hasattr(camera, 'frame_buffer') and camera.frame_buffer is not None:
                    cv2.imwrite(filename, camera.frame_buffer)
                    print(f"📸 Snapshot saved: {filename}")
                    
                    self.alert_system.trigger_alert(
                        AlertLevel.INFO,
                        f"Snapshot saved: {filename}",
                        AlertType.VISUAL
                    )
                    return True
            
            print("⚠️ Cannot take snapshot - no frame buffer")
            return False
            
        except Exception as e:
            print(f"❌ Error taking snapshot: {e}")
            return False
    
    def _calibrate_system(self):
        """Hiệu chuẩn hệ thống"""
        print("🔧 Calibrating system...")
        
        self.alert_system.trigger_alert(
            AlertLevel.INFO,
            "System calibration in progress...",
            AlertType.VISUAL
        )
        
        # Reset enhancer settings
        if hasattr(self.config, 'GAMMA_VALUE'):
            self.enhancer.gamma = self.config.GAMMA_VALUE
        if hasattr(self.config, 'USE_GAMMA'):
            self.enhancer.use_gamma = self.config.USE_GAMMA
        if hasattr(self.config, 'USE_HIST_EQ'):
            self.enhancer.use_hist = self.config.USE_HIST_EQ
        
        self.alert_system.trigger_alert(
            AlertLevel.INFO,
            "Calibration complete",
            AlertType.VISUAL
        )
        
        print("✅ Calibration complete")
    
    def run(self):
        """Chạy hệ thống chính"""
        try:
            # Khởi động các component
            print("🚀 Starting system components...")
            
            self.performance_monitor.start()
            print("✅ Performance monitor started")
            
            self.sensors.start()
            print("✅ Sensors started")
            
            if self.config.RECORD_VIDEO:
                self.recorder.start()
                print("✅ Recording started")
            
            self.running = True
            
            # Tạo window
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, 
                           self.config.FRAME_WIDTH, 
                           self.config.FRAME_HEIGHT)
            
            print("\n🎮 CONTROLS:")
            print("   Q: Quit")
            print("   S: Snapshot")
            print("   R: Toggle recording")
            print("   H: Help")
            print("   D: Cycle display modes")
            print("   F: Toggle fullscreen")
            print("   E: Toggle enhancement")
            print("   C: Calibrate")
            print("=" * 60)
            
            last_status_time = time.time()
            
            # Main loop
            while self.running:
                try:
                    # 1. Đọc frame từ camera
                    if self.active_camera in self.camera_system.cameras:
                        frame, frame_status = self.camera_system.get_active_frame()
                    else:
                        print("⚠️ No active camera")
                        time.sleep(0.1)
                        continue
                    
                    if frame is None:
                        print("⚠️ No frame from camera, retrying...")
                        time.sleep(0.1)
                        continue
                    
                    self.frame_count += 1
                    
                    # 2. Xử lý frame
                    processed_frame, fps, obstacles = self._process_frame(frame)
                    
                    if processed_frame is None:
                        continue
                    
                    # 3. Lấy camera info
                    camera_info = None
                    if self.active_camera in self.camera_system.cameras:
                        camera = self.camera_system.cameras[self.active_camera]
                        if hasattr(camera, 'get_info'):
                            camera_info = camera.get_info()
                    
                    # 4. Vẽ dashboard
                    display_frame = self.dashboard.draw(
                        processed_frame, 
                        fps, 
                        self.recorder.recording,
                        camera_info
                    )
                    
                    # 5. Ghi video
                    if self.recorder.recording and not self.recorder.paused:
                        self.recorder.write(display_frame)
                    
                    # 6. Hiển thị
                    cv2.imshow(self.window_name, display_frame)
                    
                    # 7. Xử lý phím bấm
                    key = cv2.waitKey(1) & 0xFF
                    if key != 255:
                        self._handle_keypress(key)
                    
                    # 8. Cập nhật alerts
                    self.alert_system.update_alerts()
                    
                    # 9. In status định kỳ
                    current_time = time.time()
                    if current_time - last_status_time > 5.0:  # Mỗi 5 giây
                        elapsed = current_time - self.start_time
                        fps = self.frame_count / elapsed if elapsed > 0 else 0
                        print(f"\r📊 FPS: {fps:.1f} | Frames: {self.frame_count} | Alerts: {len(self.alert_system.active_alerts)}", end="")
                        last_status_time = current_time
                    
                except Exception as e:
                    print(f"\n❌ Error in main loop: {e}")
                    traceback.print_exc()
                    time.sleep(1.0)
        
        except KeyboardInterrupt:
            print("\n\n🛑 Interrupted by user")
        
        except Exception as e:
            print(f"\n❌ Fatal error: {e}")
            traceback.print_exc()
        
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Tắt hệ thống"""
        print("\n" + "=" * 60)
        print("SHUTTING DOWN SYSTEM...")
        print("=" * 60)
        
        # Dừng các component
        self.running = False
        
        # Performance monitor
        self.performance_monitor.stop()
        
        # Sensors
        self.sensors.stop()
        
        # Recorder
        if hasattr(self.recorder, 'recording') and self.recorder.recording:
            self.recorder.stop()
        if hasattr(self.recorder, 'release'):
            self.recorder.release()
        
        # Camera system
        self.camera_system.release_all()
        
        # Alert system
        if hasattr(self.alert_system, 'release'):
            self.alert_system.release()
        
        # Log final stats
        elapsed = time.time() - self.start_time
        avg_fps = self.frame_count / elapsed if elapsed > 0 else 0
        
        print(f"\n📈 FINAL STATISTICS")
        print(f"   Total runtime: {elapsed:.1f}s")
        print(f"   Total frames: {self.frame_count}")
        print(f"   Average FPS: {avg_fps:.1f}")
        print("=" * 60)
        print("👋 System shutdown complete. Goodbye!")
        print("=" * 60)
        
        # Đóng window
        cv2.destroyAllWindows()


def main():
    """Hàm main"""
    try:
        # Tạo và chạy hệ thống
        system = ObstacleDetectionSystem(config)
        system.run()
        
    except Exception as e:
        print(f"❌ Application error: {e}")
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())