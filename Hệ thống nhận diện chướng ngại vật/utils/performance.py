# utils/performance.py - Phiên bản đã sửa lỗi import
import time
import threading
from collections import deque

# Import với try-except để tránh lỗi
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠️ psutil not available for performance monitoring")

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("⚠️ GPUtil not available for GPU monitoring")

class PerformanceMonitor:
    def __init__(self, config):
        self.config = config
        self.running = False
        self.monitor_thread = None
        
        # Metrics
        self.metrics = {
            'fps': deque(maxlen=100),
            'cpu_usage': deque(maxlen=100),
            'memory_usage': deque(maxlen=100),
            'gpu_usage': deque(maxlen=100),
            'gpu_temp': deque(maxlen=100),
            'frame_process_time': deque(maxlen=100),
            'network_latency': deque(maxlen=100),
        }
        
        # Alerts thresholds
        self.thresholds = {
            'fps_low': 15,
            'cpu_high': 85,
            'memory_high': 85,
            'gpu_temp_high': 85,
            'frame_time_high': 100,  # ms
        }
        
        # Performance states
        self.states = {
            'fps_state': 'normal',  # normal/low
            'cpu_state': 'normal',  # normal/high
            'memory_state': 'normal',  # normal/high
            'gpu_state': 'normal',  # normal/hot
            'overall_state': 'normal',
        }
        
        # Adaptive settings
        self.adaptive_settings = {
            'resolution_scale': 1.0,
            'enhancement_level': 'high',
            'processing_mode': 'normal',
        }
        
        # Callbacks
        self.alert_callbacks = []
        self.adjustment_callbacks = []
        
        # Update interval
        self.update_interval = 2.0  # giây
    
    def _update_metrics(self):
        """Cập nhật các metrics"""
        # CPU usage
        if PSUTIL_AVAILABLE:
            cpu_percent = psutil.cpu_percent(interval=0.1)
        else:
            import random
            cpu_percent = 20 + random.random() * 30  # Simulated
        
        self.metrics['cpu_usage'].append(cpu_percent)
        
        # Memory usage
        if PSUTIL_AVAILABLE:
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
        else:
            import random
            memory_percent = 40 + random.random() * 30  # Simulated
        
        self.metrics['memory_usage'].append(memory_percent)
        
        # GPU metrics
        if GPU_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    self.metrics['gpu_usage'].append(gpu.load * 100)
                    self.metrics['gpu_temp'].append(gpu.temperature)
                else:
                    self.metrics['gpu_usage'].append(0)
                    self.metrics['gpu_temp'].append(40)
            except:
                self.metrics['gpu_usage'].append(0)
                self.metrics['gpu_temp'].append(40)
        else:
            # Simulated GPU stats
            import random
            self.metrics['gpu_usage'].append(15 + random.random() * 20)
            self.metrics['gpu_temp'].append(40 + random.random() * 10)
        
        # Network latency (simulated)
        self.metrics['network_latency'].append(10 + cpu_percent * 0.1)
    
    # ... (giữ nguyên các method khác) ...

    def update_frame_metrics(self, fps, process_time_ms):
        """Cập nhật metrics về frame"""
        self.metrics['fps'].append(fps)
        self.metrics['frame_process_time'].append(process_time_ms)
    
    def _check_thresholds(self):
        """Kiểm tra ngưỡng và cập nhật trạng thái"""
        # FPS state
        if self.metrics['fps']:
            avg_fps = sum(self.metrics['fps']) / len(self.metrics['fps'])
            if avg_fps < self.thresholds['fps_low']:
                new_state = 'low'
            else:
                new_state = 'normal'
            
            if new_state != self.states['fps_state']:
                self.states['fps_state'] = new_state
                if new_state == 'low':
                    self._trigger_alert('fps_low', 
                                      f"Low FPS detected: {avg_fps:.1f}")
        
        # CPU state
        if self.metrics['cpu_usage']:
            avg_cpu = sum(self.metrics['cpu_usage']) / len(self.metrics['cpu_usage'])
            if avg_cpu > self.thresholds['cpu_high']:
                new_state = 'high'
            else:
                new_state = 'normal'
            
            if new_state != self.states['cpu_state']:
                self.states['cpu_state'] = new_state
                if new_state == 'high':
                    self._trigger_alert('cpu_high',
                                      f"High CPU usage: {avg_cpu:.1f}%")
        
        # Memory state
        if self.metrics['memory_usage']:
            avg_memory = sum(self.metrics['memory_usage']) / len(self.metrics['memory_usage'])
            if avg_memory > self.thresholds['memory_high']:
                new_state = 'high'
            else:
                new_state = 'normal'
            
            if new_state != self.states['memory_state']:
                self.states['memory_state'] = new_state
                if new_state == 'high':
                    self._trigger_alert('memory_high',
                                      f"High memory usage: {avg_memory:.1f}%")
        
        # GPU temperature
        if self.metrics['gpu_temp']:
            avg_temp = sum(self.metrics['gpu_temp']) / len(self.metrics['gpu_temp'])
            if avg_temp > self.thresholds['gpu_temp_high']:
                new_state = 'hot'
            else:
                new_state = 'normal'
            
            if new_state != self.states['gpu_state']:
                self.states['gpu_state'] = new_state
                if new_state == 'hot':
                    self._trigger_alert('gpu_hot',
                                      f"High GPU temperature: {avg_temp:.1f}°C")
        
        # Update overall state
        self._update_overall_state()
    
    def _update_overall_state(self):
        """Cập nhật trạng thái tổng thể"""
        states = list(self.states.values())
        
        if 'hot' in states:
            overall = 'critical'
        elif 'high' in states:
            overall = 'warning'
        elif 'low' in states:
            overall = 'degraded'
        else:
            overall = 'normal'
        
        self.states['overall_state'] = overall
    
    def _trigger_alert(self, alert_type, message):
        """Kích hoạt cảnh báo performance"""
        for callback in self.alert_callbacks:
            try:
                callback(alert_type, message, self.get_metrics_summary())
            except Exception as e:
                print(f"❌ Performance alert callback error: {e}")
    
    def _adjust_settings(self):
        """Điều chỉnh cài đặt dựa trên performance"""
        # Kiểm tra nếu cần điều chỉnh
        if self.states['overall_state'] == 'critical':
            # Critical: Giảm resolution và tắt enhancement
            new_settings = {
                'resolution_scale': 0.5,
                'enhancement_level': 'low',
                'processing_mode': 'fast',
            }
        elif self.states['overall_state'] == 'warning':
            # Warning: Giảm nhẹ quality
            new_settings = {
                'resolution_scale': 0.75,
                'enhancement_level': 'medium',
                'processing_mode': 'balanced',
            }
        elif self.states['overall_state'] == 'degraded':
            # Degraded: Tối ưu cho FPS
            new_settings = {
                'resolution_scale': 0.9,
                'enhancement_level': 'medium',
                'processing_mode': 'balanced',
            }
        else:
            # Normal: Sử dụng cài đặt cao nhất
            new_settings = {
                'resolution_scale': 1.0,
                'enhancement_level': 'high',
                'processing_mode': 'quality',
            }
        
        # Kiểm tra xem có thay đổi không
        if new_settings != self.adaptive_settings:
            self.adaptive_settings = new_settings
            
            # Gọi callbacks
            for callback in self.adjustment_callbacks:
                try:
                    callback(new_settings)
                except Exception as e:
                    print(f"❌ Performance adjustment callback error: {e}")
            
            print(f"🔄 Performance settings adjusted: {new_settings}")
    
    def start(self):
        """Bắt đầu monitoring"""
        if self.running:
            return True
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        print("📊 Performance monitor started")
        return True
    
    def stop(self):
        """Dừng monitoring"""
        self.running = False
        if self.monitor_thread is not None:
            self.monitor_thread.join(timeout=2.0)
        print("📊 Performance monitor stopped")
    
    def _monitor_loop(self):
        """Vòng lặp monitoring"""
        while self.running:
            try:
                self._update_metrics()
                self._check_thresholds()
                self._adjust_settings()
                time.sleep(self.update_interval)
            except Exception as e:
                print(f"❌ Performance monitor error: {e}")
                time.sleep(5.0)
    
    def get_metrics_summary(self):
        """Lấy tổng quan metrics"""
        summary = {}
        
        for metric_name, values in self.metrics.items():
            if values:
                summary[metric_name] = {
                    'current': values[-1] if values else 0,
                    'average': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'trend': 'up' if len(values) > 1 and values[-1] > values[-2] else 'down'
                }
        
        summary['states'] = self.states.copy()
        summary['adaptive_settings'] = self.adaptive_settings.copy()
        
        return summary
    
    def get_current_state(self):
        """Lấy trạng thái hiện tại"""
        return {
            'states': self.states.copy(),
            'settings': self.adaptive_settings.copy(),
            'fps': self.metrics['fps'][-1] if self.metrics['fps'] else 0,
            'cpu': self.metrics['cpu_usage'][-1] if self.metrics['cpu_usage'] else 0,
            'memory': self.metrics['memory_usage'][-1] if self.metrics['memory_usage'] else 0,
        }
    
    def register_alert_callback(self, callback):
        """Đăng ký callback cho alert"""
        if callback not in self.alert_callbacks:
            self.alert_callbacks.append(callback)
    
    def register_adjustment_callback(self, callback):
        """Đăng ký callback cho adjustment"""
        if callback not in self.adjustment_callbacks:
            self.adjustment_callbacks.append(callback)