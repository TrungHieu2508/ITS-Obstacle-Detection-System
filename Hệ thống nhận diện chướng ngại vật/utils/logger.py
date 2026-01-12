# utils/logger.py - Hệ thống logging nâng cao
import logging
import logging.handlers
import json
import os
from datetime import datetime
import threading

class StructuredLogger:
    def __init__(self, config):
        self.config = config
        self.logger = None
        self.log_file = None
        self.performance_log = None
        self.event_log = None
        
        # Buffer cho high-frequency logs
        self.log_buffer = []
        self.buffer_lock = threading.Lock()
        self.buffer_size = 100
        self.flush_interval = 5  # giây
        
        # Khởi tạo logging
        self._setup_logging()
        
        # Start flush thread
        self.flush_thread = threading.Thread(target=self._flush_buffer_loop)
        self.flush_thread.daemon = True
        self.flush_thread.start()
    
    def _setup_logging(self):
        """Thiết lập hệ thống logging"""
        # Tạo logger
        self.logger = logging.getLogger('ObstacleDetection')
        self.logger.setLevel(getattr(logging, self.config.LOG_LEVEL))
        
        # Tạo thư mục logs nếu chưa có
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        
        # File handler
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"system_{timestamp}.log")
        
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Thêm handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Tạo file log cho performance
        self.performance_log = os.path.join(log_dir, f"performance_{timestamp}.jsonl")
        
        # Tạo file log cho events
        self.event_log = os.path.join(log_dir, f"events_{timestamp}.jsonl")
        
        print(f"📝 Logging initialized: {self.log_file}")
    
    def _flush_buffer_loop(self):
        """Vòng lặp flush buffer định kỳ"""
        while True:
            time.sleep(self.flush_interval)
            self.flush_buffer()
    
    def log(self, level, message, **kwargs):
        """Ghi log với structured data"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'message': message,
            **kwargs
        }
        
        # Thêm vào buffer
        with self.buffer_lock:
            self.log_buffer.append(log_entry)
            
            # Nếu buffer đầy, flush
            if len(self.log_buffer) >= self.buffer_size:
                self._flush_buffer()
    
    def _flush_buffer(self):
        """Flush buffer ra file"""
        with self.buffer_lock:
            if not self.log_buffer:
                return
            
            # Ghi ra performance log (JSON Lines format)
            with open(self.performance_log, 'a') as f:
                for entry in self.log_buffer:
                    f.write(json.dumps(entry) + '\n')
            
            # Clear buffer
            self.log_buffer.clear()
    
    def flush_buffer(self):
        """Flush buffer (public method)"""
        self._flush_buffer()
    
    def log_performance(self, fps, cpu_usage, memory_usage, **kwargs):
        """Ghi log performance"""
        self.log(
            'PERFORMANCE',
            'System performance metrics',
            fps=fps,
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            **kwargs
        )
    
    def log_event(self, event_type, description, severity='INFO', **kwargs):
        """Ghi log sự kiện"""
        event_entry = {
            'timestamp': datetime.now().isoformat(),
            'type': event_type,
            'description': description,
            'severity': severity,
            **kwargs
        }
        
        # Ghi ra event log
        with open(self.event_log, 'a') as f:
            f.write(json.dumps(event_entry) + '\n')
        
        # Cũng ghi vào system log
        self.log(severity, f"Event: {event_type} - {description}", **kwargs)
    
    def log_alert(self, alert_data):
        """Ghi log cảnh báo"""
        self.log_event(
            'ALERT',
            alert_data.get('message', 'Unknown alert'),
            severity=alert_data.get('level', 'INFO').upper(),
            alert_data=alert_data
        )
    
    def log_obstacle(self, obstacles):
        """Ghi log chướng ngại vật"""
        if obstacles:
            self.log(
                'INFO',
                f"Detected {len(obstacles)} obstacles",
                obstacles=obstacles,
                obstacle_count=len(obstacles)
            )
    
    def log_camera_status(self, camera_info):
        """Ghi log trạng thái camera"""
        self.log(
            'INFO',
            f"Camera status update",
            camera_id=camera_info.get('id'),
            connected=camera_info.get('connected'),
            resolution=camera_info.get('resolution'),
            fps=camera_info.get('fps'),
            drop_rate=camera_info.get('drop_rate', 0)
        )
    
    def get_log_stats(self):
        """Lấy thống kê log"""
        stats = {
            'buffer_size': len(self.log_buffer),
            'log_file': self.log_file,
            'performance_log': self.performance_log,
            'event_log': self.event_log,
            'last_flush': datetime.now().isoformat()
        }
        
        # Thêm file sizes
        for filepath in [self.log_file, self.performance_log, self.event_log]:
            if os.path.exists(filepath):
                stats[os.path.basename(filepath) + '_size'] = \
                    os.path.getsize(filepath)
        
        return stats
    
    def cleanup_old_logs(self, days_to_keep=7):
        """Dọn dẹp log cũ"""
        log_dir = "logs"
        if not os.path.exists(log_dir):
            return
        
        cutoff_time = datetime.now().timestamp() - (days_to_keep * 24 * 3600)
        
        for filename in os.listdir(log_dir):
            filepath = os.path.join(log_dir, filename)
            if os.path.isfile(filepath):
                if os.path.getmtime(filepath) < cutoff_time:
                    try:
                        os.remove(filepath)
                        self.log('INFO', f"Removed old log file: {filename}")
                    except Exception as e:
                        self.log('ERROR', f"Failed to remove old log: {e}")
    
    def debug(self, message, **kwargs):
        self.log('DEBUG', message, **kwargs)
    
    def info(self, message, **kwargs):
        self.log('INFO', message, **kwargs)
    
    def warning(self, message, **kwargs):
        self.log('WARNING', message, **kwargs)
    
    def error(self, message, **kwargs):
        self.log('ERROR', message, **kwargs)
    
    def critical(self, message, **kwargs):
        self.log('CRITICAL', message, **kwargs)