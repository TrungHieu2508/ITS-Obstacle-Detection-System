# main_stage2.py - Hệ thống phát hiện chướng ngại vật nâng cao
import cv2
import time
import sys
import traceback
import numpy as np
from datetime import datetime

import config
from core.camera import MultiCameraSystem, Camera
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
        
        # Khởi tạo các component
        self.logger = StructuredLogger(config)
        self.performance_monitor = PerformanceMonitor(config)
        self.alert_system = AlertSystem(config)
        
        # Camera system
        self.camera_system = MultiCameraSystem(config.CAMERA_SOURCES)
        self.active_camera = 'primary'
        
        # Image processing
        self.enhancer = ImageEnhancer(config)
        self.fps_counter = FPSCounter(config.FPS_BUFFER_SIZE)
        
        # Display and UI
        self.dashboard = Dashboard(config)
        self.fullscreen = False
        self.window_name = "🚗 Obstacle Detection System v2.0"
        
        # Recording
        camera_info = self.camera_system.cameras.get(self.active_camera)
        if camera_info:
            self.recorder = VideoRecorder(config, camera_info.get_info())
        else:
            self.recorder = VideoRecorder(config, {})
        
        # Sensors
        self.sensors = VehicleSensors(config)
        
        # Obstacle detection (placeholder)
        self.obstacle_detector = self._create_obstacle_detector()
        
        # Register callbacks
        self._register_callbacks()
        
        # Statistics
        self.frame_count = 0
        self.start_time = time.time()
        self.last_snapshot_time = 0
        
        # PIP settings
        self.show_pip = config.SHOW_PIP
        self.pip_camera = 'secondary' if 'secondary' in config.CAMERA_SOURCES else 'primary'
        
        print("=" * 60)
        print("🚗 OBSTACLE DETECTION SYSTEM - STAGE 2")
        print("=" * 60)
        print(f"📊 Resolution: {config.FRAME_WIDTH}x{config.FRAME_HEIGHT}")
        print(f"🎥 Cameras: {list(config.CAMERA_SOURCES.keys())}")
        print(f"💾 Recording: {'ON' if config.RECORD_VIDEO else 'OFF'}")
        print(f"🚨 Alerts: Audio={config.ENABLE_AUDIO_ALERTS}, Visual={config.ENABLE_VISUAL_ALERTS}")
        print("=" * 60)
    
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
        
        # Alert system logging
        self.alert_system.trigger_alert = self._wrap_alert_trigger(
            self.alert_system.trigger_alert
        )
    
    def _wrap_alert_trigger(self, original_func):
        """Wrapper để log alerts"""
        def wrapped(*args, **kwargs):
            result = original_func(*args, **kwargs)
            if result:
                alert = self.alert_system.active_alerts.get(result)
                if alert:
                    self.logger.log_alert(alert)
            return result
        return wrapped
    
    def _handle_performance_adjustment(self, settings):
        """Xử lý điều chỉnh performance"""
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
    
    def _handle_sensor_data(self, sensor_data):
        """Xử lý dữ liệu cảm biến"""
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
    
    def _check_sensor_alerts(self, sensor_data):
        """Kiểm tra cảnh báo từ cảm biến"""
        vehicle_data = sensor_data['vehicle']
        gps_data = sensor_data['gps']
        
        # Kiểm tra tốc độ
        if gps_data['speed'] > 100:  # >100 km/h
            self.alert_system.trigger_alert(
                AlertLevel.MEDIUM,
                f"High speed: {gps_data['speed']:.0f} km/h",
                AlertType.ALL
            )
        
        # Kiểm tra nhiệt độ động cơ
        if vehicle_data['engine_temp'] > 100:
            self.alert_system.trigger_alert(
                AlertLevel.HIGH,
                f"Engine overheating: {vehicle_data['engine_temp']:.1f}°C",
                AlertType.ALL
            )
        
        # Kiểm tra nhiên liệu
        if vehicle_data['fuel_level'] < 15:
            self.alert_system.trigger_alert(
                AlertLevel.LOW,
                f"Low fuel: {vehicle_data['fuel_level']:.1f}%",
                AlertType.ALL
            )
        
        # Kiểm tra áp suất lốp
        for tire, pressure in vehicle_data['tire_pressure'].items():
            if pressure < 2.0:
                self.alert_system.trigger_alert(
                    AlertLevel.MEDIUM,
                    f"Low tire pressure ({tire}): {pressure:.1f} bar",
                    AlertType.ALL
                )
    
    def _draw_pip(self, frame):
        """Vẽ Picture-in-Picture"""
        if not self.show_pip or self.pip_camera == self.active_camera:
            return frame
        
        pip_frame = self.camera_system.get_pip_frame(
            self.pip_camera, 
            size=self.config.PIP_SIZE
        )
        
        if pip_frame is not None:
            h, w = frame.shape[:2]
            pip_w, pip_h = self.config.PIP_SIZE
            
            # Vị trí PIP (góc trên phải)
            pip_x = w - pip_w - 20
            pip_y = 20
            
            # Border cho PIP
            cv2.rectangle(frame, (pip_x-2, pip_y-2), 
                         (pip_x+pip_w+2, pip_y+pip_h+2),
                         (255, 255, 255), 1)
            
            # Thêm PIP frame
            frame[pip_y:pip_y+pip_h, pip_x:pip_x+pip_w] = pip_frame
            
            # Label
            cv2.putText(frame, f"CAM: {self.pip_camera.upper()}", 
                       (pip_x + 5, pip_y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def _process_frame(self, frame):
        """Xử lý một frame"""
        if frame is None:
            return None
        
        start_process_time = time.time() * 1000
        
        # 1. Enhancement
        enhanced_frame = self.enhancer.process(frame)
        
        # 2. Obstacle detection
        obstacles = self.obstacle_detector.detect(enhanced_frame)
        
        # 3. Cập nhật dashboard với obstacles
        self.dashboard.update_obstacles(obstacles)
        
        # 4. Kiểm tra cảnh báo obstacles
        self._check_obstacle_alerts(obstacles)
        
        # 5. Log obstacles
        self.logger.log_obstacle(obstacles)
        
        # 6. Tính FPS
        fps_avg = self.fps_counter.update()
        self.dashboard.update_fps(fps_avg)
        
        # 7. Cập nhật performance metrics
        process_time = time.time() * 1000 - start_process_time
        self.performance_monitor.update_frame_metrics(fps_avg, process_time)
        
        return enhanced_frame, fps_avg, obstacles
    
    def _check_obstacle_alerts(self, obstacles):
        """Kiểm tra cảnh báo từ chướng ngại vật"""
        if not obstacles:
            return
        
        # Phân loại obstacles theo confidence và distance
        high_conf_obstacles = [o for o in obstacles if o['confidence'] > 0.7]
        close_obstacles = [o for o in obstacles if o.get('distance', 100) < 5]
        
        if close_obstacles:
            # Obstacle gần -> cảnh báo cao
            self.alert_system.trigger_alert(
                AlertLevel.HIGH,
                f"Close obstacle detected! Distance: {close_obstacles[0]['distance']:.1f}m",
                AlertType.ALL,
                position=close_obstacles[0].get('position')
            )
            
            # Trigger event recording
            if self.config.RECORD_EVENTS_ONLY:
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
            current_index = cameras.index(self.active_camera)
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
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"snapshot_{timestamp}.jpg"
        
        # Lấy frame gần nhất từ camera active
        camera = self.camera_system.cameras.get(self.active_camera)
        if camera and camera.frame_buffer is not None:
            cv2.imwrite(filename, camera.frame_buffer)
            print(f"📸 Snapshot saved: {filename}")
            
            self.alert_system.trigger_alert(
                AlertLevel.INFO,
                f"Snapshot saved: {filename}",
                AlertType.VISUAL
            )
            
            return True
        
        return False
    
    def _calibrate_system(self):
        """Hiệu chuẩn hệ thống"""
        print("🔧 Calibrating system...")
        
        self.alert_system.trigger_alert(
            AlertLevel.INFO,
            "System calibration in progress...",
            AlertType.VISUAL
        )
        
        # Calibrate camera (nếu có)
        camera = self.camera_system.cameras.get(self.active_camera)
        if camera:
            # Điều chỉnh dựa trên ánh sáng hiện tại
            if camera.frame_buffer is not None:
                light_level = self.enhancer.estimate_light_level(camera.frame_buffer)
                camera.adjust_for_lighting(light_level)
        
        # Reset enhancer settings
        self.enhancer.gamma = self.config.GAMMA_VALUE
        self.enhancer.use_gamma = self.config.USE_GAMMA
        self.enhancer.use_hist = self.config.USE_HIST_EQ
        
        self.alert_system.trigger_alert(
            AlertLevel.INFO,
            "Calibration complete",
            AlertType.VISUAL
        )
        
        print("✅ Calibration complete")
    
    def _print_status(self):
        """In trạng thái hệ thống"""
        elapsed = time.time() - self.start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0
        
        print("\n" + "=" * 60)
        print("SYSTEM STATUS")
        print("=" * 60)
        print(f"🕒 Uptime: {elapsed:.1f}s")
        print(f"📊 FPS: {fps:.1f} (Current: {self.fps_counter.buffer[-1] if self.fps_counter.buffer else 0:.1f})")
        print(f"🎥 Active camera: {self.active_camera}")
        print(f"💾 Recording: {'ON' if self.recorder.recording else 'OFF'}")
        print(f"🚨 Alerts active: {len(self.alert_system.active_alerts)}")
        print(f"📈 Performance: {self.performance_monitor.states['overall_state']}")
        print("=" * 60)
    
    def run(self):
        """Chạy hệ thống chính"""
        try:
            # Khởi động các component
            self.performance_monitor.start()
            self.sensors.start()
            
            if self.config.RECORD_VIDEO:
                self.recorder.start()
            
            self.running = True
            
            # Tạo window
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, 
                           self.config.FRAME_WIDTH, 
                           self.config.FRAME_HEIGHT)
            
            print("🚀 System started. Press:")
            print("   Q: Quit")
            print("   S: Snapshot")
            print("   R: Toggle recording")
            print("   H: Help")
            print("   D: Cycle display modes")
            print("   F: Toggle fullscreen")
            
            last_status_time = time.time()
            
            # Main loop
            while self.running:
                try:
                    # 1. Đọc frame từ camera
                    frame, frame_status = self.camera_system.get_active_frame()
                    
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
                    active_cam = self.camera_system.cameras.get(self.active_camera)
                    if active_cam:
                        camera_info = active_cam.get_info()
                    
                    # 4. Vẽ dashboard
                    display_frame = self.dashboard.draw(
                        processed_frame, 
                        fps, 
                        self.recorder.recording,
                        camera_info
                    )
                    
                    # 5. Thêm PIP
                    if self.show_pip:
                        display_frame = self._draw_pip(display_frame)
                    
                    # 6. Ghi video
                    if self.recorder.recording and not self.recorder.paused:
                        self.recorder.write(display_frame)
                    
                    # 7. Hiển thị
                    cv2.imshow(self.window_name, display_frame)
                    
                    # 8. Xử lý phím bấm
                    key = cv2.waitKey(1) & 0xFF
                    if key != 255:
                        self._handle_keypress(key)
                    
                    # 9. Cập nhật alerts
                    self.alert_system.update_alerts()
                    
                    # 10. Log định kỳ
                    if self.frame_count % 100 == 0:
                        self.logger.log_performance(
                            fps,
                            self.performance_monitor.metrics['cpu_usage'][-1] if self.performance_monitor.metrics['cpu_usage'] else 0,
                            self.performance_monitor.metrics['memory_usage'][-1] if self.performance_monitor.metrics['memory_usage'] else 0,
                            frame_count=self.frame_count,
                            active_alerts=len(self.alert_system.active_alerts)
                        )
                    
                    # 11. In status định kỳ
                    current_time = time.time()
                    if current_time - last_status_time > 10.0:  # Mỗi 10 giây
                        self._print_status()
                        last_status_time = current_time
                    
                except Exception as e:
                    print(f"❌ Error in main loop: {e}")
                    traceback.print_exc()
                    time.sleep(1.0)
        
        except KeyboardInterrupt:
            print("\n\n🛑 Interrupted by user")
        
        except Exception as e:
            print(f"❌ Fatal error: {e}")
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
        if self.recorder.recording:
            self.recorder.stop()
        self.recorder.release()
        
        # Camera system
        self.camera_system.release_all()
        
        # Alert system
        self.alert_system.release()
        
        # Log final stats
        elapsed = time.time() - self.start_time
        avg_fps = self.frame_count / elapsed if elapsed > 0 else 0
        
        self.logger.info(
            "System shutdown",
            uptime=elapsed,
            total_frames=self.frame_count,
            avg_fps=avg_fps,
            total_alerts=self.alert_system.stats['total_alerts']
        )
        
        # Flush logs
        self.logger.flush_buffer()
        
        # Đóng window
        cv2.destroyAllWindows()
        
        # Print final stats
        print(f"\n📈 FINAL STATISTICS")
        print(f"   Total runtime: {elapsed:.1f}s")
        print(f"   Total frames: {self.frame_count}")
        print(f"   Average FPS: {avg_fps:.1f}")
        print(f"   Total alerts: {self.alert_system.stats['total_alerts']}")
        print(f"   Log file: {self.logger.log_file}")
        print("=" * 60)
        print("👋 System shutdown complete. Goodbye!")
        print("=" * 60)


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