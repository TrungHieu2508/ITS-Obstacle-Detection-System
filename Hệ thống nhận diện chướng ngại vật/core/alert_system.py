# core/alert_system.py - Hệ thống cảnh báo đa phương thức
import threading
import time
import winsound  # Chỉ trên Windows
import numpy as np
from datetime import datetime
from enum import Enum

class AlertLevel(Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class AlertType(Enum):
    VISUAL = "visual"
    AUDIO = "audio"
    HAPTIC = "haptic"
    ALL = "all"

class AlertSystem:
    def __init__(self, config):
        self.config = config
        self.enabled = True
        
        # Các cảnh báo đang active
        self.active_alerts = {}
        self.alert_history = []
        self.max_history = 100
        
        # Cấu hình cảnh báo
        self.alert_configs = {
            AlertLevel.LOW: {
                'color': config.COLOR_WARNING,
                'sound_freq': 440,
                'sound_duration': 0.3,
                'repeat_interval': 2.0,
                'priority': 1
            },
            AlertLevel.MEDIUM: {
                'color': config.COLOR_WARNING,
                'sound_freq': 660,
                'sound_duration': 0.5,
                'repeat_interval': 1.0,
                'priority': 2
            },
            AlertLevel.HIGH: {
                'color': config.COLOR_DANGER,
                'sound_freq': 880,
                'sound_duration': 1.0,
                'repeat_interval': 0.5,
                'priority': 3
            },
            AlertLevel.INFO: {
                'color': (100, 200, 255),  # Xanh nhạt
                'sound_freq': 330,
                'sound_duration': 0.1,
                'repeat_interval': 0,
                'priority': 0
            }
        }
        
        # Thread cho âm thanh
        self.audio_thread = None
        self.audio_queue = []
        self.audio_lock = threading.Lock()
        self.audio_stop = threading.Event()
        
        # Haptic feedback (giả lập)
        self.haptic_enabled = False
        self.haptic_patterns = {
            AlertLevel.LOW: [100, 200, 100],  # ms
            AlertLevel.MEDIUM: [100, 100, 100],
            AlertLevel.HIGH: [50, 50, 50, 50, 50]
        }
        
        # Statistics
        self.stats = {
            'total_alerts': 0,
            'active_alerts': 0,
            'by_level': {level.value: 0 for level in AlertLevel},
            'by_type': {atype.value: 0 for atype in AlertType}
        }
        
        # Khởi động audio thread
        self._start_audio_thread()
    
    def _start_audio_thread(self):
        """Khởi động thread xử lý âm thanh"""
        if self.config.ENABLE_AUDIO_ALERTS:
            self.audio_stop.clear()
            self.audio_thread = threading.Thread(target=self._audio_worker)
            self.audio_thread.daemon = True
            self.audio_thread.start()
            print("🔊 Audio alert system started")
    
    def _audio_worker(self):
        """Worker thread cho âm thanh"""
        while not self.audio_stop.is_set():
            with self.audio_lock:
                if self.audio_queue:
                    alert_id, sound_config = self.audio_queue.pop(0)
                    
                    # Kiểm tra xem alert còn active không
                    if alert_id in self.active_alerts:
                        try:
                            # Phát âm thanh
                            duration_ms = int(sound_config['sound_duration'] * 1000)
                            winsound.Beep(sound_config['sound_freq'], duration_ms)
                        except Exception as e:
                            print(f"❌ Lỗi phát âm thanh: {e}")
                    
                    # Nếu cần lặp lại, thêm lại vào queue
                    if alert_id in self.active_alerts:
                        repeat_interval = sound_config.get('repeat_interval', 0)
                        if repeat_interval > 0:
                            # Thêm lại sau interval
                            threading.Timer(
                                repeat_interval,
                                lambda: self._add_to_audio_queue(alert_id, sound_config)
                            ).start()
            
            time.sleep(0.1)
    
    def _add_to_audio_queue(self, alert_id, sound_config):
        """Thêm cảnh báo vào audio queue"""
        with self.audio_lock:
            self.audio_queue.append((alert_id, sound_config))
    
    def trigger_alert(self, level, message, alert_type=AlertType.ALL, 
                     position=None, duration=5.0, data=None):
        """Kích hoạt cảnh báo mới"""
        if not self.enabled:
            return None
        
        alert_id = f"{level.value}_{int(time.time() * 1000)}"
        
        alert_config = self.alert_configs.get(level, self.alert_configs[AlertLevel.INFO])
        
        alert = {
            'id': alert_id,
            'level': level,
            'message': message,
            'type': alert_type,
            'position': position,
            'timestamp': datetime.now(),
            'duration': duration,
            'config': alert_config,
            'data': data or {},
            'active': True
        }
        
        # Thêm vào active alerts
        self.active_alerts[alert_id] = alert
        
        # Thêm vào history
        self.alert_history.append(alert.copy())
        if len(self.alert_history) > self.max_history:
            self.alert_history.pop(0)
        
        # Cập nhật statistics
        self.stats['total_alerts'] += 1
        self.stats['active_alerts'] = len(self.active_alerts)
        self.stats['by_level'][level.value] += 1
        self.stats['by_type'][alert_type.value] += 1
        
        # Kích hoạt các loại cảnh báo
        self._activate_alert_types(alert_id, alert)
        
        print(f"🚨 Alert triggered: [{level.value}] {message}")
        
        return alert_id
    
    def _activate_alert_types(self, alert_id, alert):
        """Kích hoạt các loại cảnh báo cụ thể"""
        alert_type = alert['type']
        level = alert['level']
        config = alert['config']
        
        # Visual alert (luôn active)
        if alert_type in [AlertType.VISUAL, AlertType.ALL]:
            pass  # Được xử lý bởi dashboard
        
        # Audio alert
        if self.config.ENABLE_AUDIO_ALERTS and alert_type in [AlertType.AUDIO, AlertType.ALL]:
            self._add_to_audio_queue(alert_id, config)
        
        # Haptic alert (giả lập)
        if self.haptic_enabled and alert_type in [AlertType.HAPTIC, AlertType.ALL]:
            self._trigger_haptic(level)
    
    def _trigger_haptic(self, level):
        """Kích hoạt haptic feedback (giả lập)"""
        if level in self.haptic_patterns:
            pattern = self.haptic_patterns[level]
            print(f"📳 Haptic pattern: {pattern}ms")
            # Trong thực tế, sẽ gọi API đến thiết bị haptic
    
    def clear_alert(self, alert_id):
        """Xóa cảnh báo"""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id]['active'] = False
            del self.active_alerts[alert_id]
            self.stats['active_alerts'] = len(self.active_alerts)
            return True
        return False
    
    def clear_all_alerts(self):
        """Xóa tất cả cảnh báo"""
        self.active_alerts.clear()
        self.stats['active_alerts'] = 0
        print("🗑️ All alerts cleared")
    
    def update_alerts(self):
        """Cập nhật trạng thái cảnh báo (xóa hết hạn)"""
        current_time = datetime.now()
        expired_alerts = []
        
        for alert_id, alert in list(self.active_alerts.items()):
            time_diff = (current_time - alert['timestamp']).total_seconds()
            if time_diff > alert['duration']:
                expired_alerts.append(alert_id)
        
        for alert_id in expired_alerts:
            self.clear_alert(alert_id)
        
        return expired_alerts
    
    def get_active_alerts(self):
        """Lấy danh sách cảnh báo đang active"""
        return list(self.active_alerts.values())
    
    def get_alert_history(self, limit=10):
        """Lấy lịch sử cảnh báo"""
        return self.alert_history[-limit:]
    
    def get_alert_summary(self):
        """Lấy tổng quan về cảnh báo"""
        level_counts = {level.value: 0 for level in AlertLevel}
        
        for alert in self.active_alerts.values():
            level_counts[alert['level'].value] += 1
        
        return {
            'total_active': len(self.active_alerts),
            'by_level': level_counts,
            'recent_count': len([a for a in self.alert_history[-10:] if a['active']])
        }
    
    def enable(self):
        """Bật hệ thống cảnh báo"""
        self.enabled = True
        print("✅ Alert system enabled")
    
    def disable(self):
        """Tắt hệ thống cảnh báo"""
        self.enabled = False
        self.clear_all_alerts()
        print("⛔ Alert system disabled")
    
    def toggle_audio(self):
        """Bật/tắt âm thanh cảnh báo"""
        if self.config.ENABLE_AUDIO_ALERTS:
            self.config.ENABLE_AUDIO_ALERTS = False
            self.audio_stop.set()
            print("🔇 Audio alerts disabled")
        else:
            self.config.ENABLE_AUDIO_ALERTS = True
            self._start_audio_thread()
            print("🔊 Audio alerts enabled")
    
    def toggle_haptic(self):
        """Bật/tắt haptic feedback"""
        self.haptic_enabled = not self.haptic_enabled
        print(f"📳 Haptic feedback {'enabled' if self.haptic_enabled else 'disabled'}")
    
    def set_alert_duration(self, level, duration):
        """Đặt thời gian hiển thị cảnh báo"""
        if level in self.alert_configs:
            self.alert_configs[level]['duration'] = duration
            return True
        return False
    
    def test_alert(self, level=AlertLevel.MEDIUM):
        """Test cảnh báo"""
        test_messages = {
            AlertLevel.INFO: "System test - Information alert",
            AlertLevel.LOW: "System test - Low priority alert",
            AlertLevel.MEDIUM: "System test - Medium priority alert",
            AlertLevel.HIGH: "System test - HIGH PRIORITY ALERT!"
        }
        
        message = test_messages.get(level, "Test alert")
        return self.trigger_alert(level, message, AlertType.ALL)
    
    def release(self):
        """Giải phóng resource"""
        self.audio_stop.set()
        if self.audio_thread is not None:
            self.audio_thread.join(timeout=2.0)
        
        print("🔇 Alert system released")