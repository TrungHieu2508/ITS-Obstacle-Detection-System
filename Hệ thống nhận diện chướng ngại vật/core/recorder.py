# core/recorder.py - Nâng cấp hệ thống ghi video thông minh
import cv2
import os
import json
import threading
import queue
from datetime import datetime
from collections import deque

class VideoRecorder:
    def __init__(self, config, camera_info):
        self.config = config
        self.camera_info = camera_info
        
        # Recording state
        self.recording = False
        self.paused = False
        self.event_recording = False
        
        # Video writer
        self.writer = None
        self.video_file = None
        self.frame_size = (config.FRAME_WIDTH, config.FRAME_HEIGHT)
        
        # Event buffer
        self.event_buffer = deque(maxlen=config.EVENT_PRE_BUFFER + config.EVENT_POST_BUFFER)
        self.event_active = False
        self.event_end_counter = 0
        
        # Metadata
        self.metadata = {
            'start_time': None,
            'duration': 0,
            'frame_count': 0,
            'events': [],
            'camera_info': camera_info,
            'system_info': {},
        }
        
        # Threading
        self.write_queue = queue.Queue(maxsize=100)
        self.writer_thread = None
        self.stop_event = threading.Event()
        
        # Statistics
        self.stats = {
            'frames_written': 0,
            'frames_dropped': 0,
            'events_recorded': 0,
            'current_bitrate': 0,
        }
        
        # Initialize
        self._setup_output_directory()
    
    def _setup_output_directory(self):
        """Thiết lập thư mục output"""
        os.makedirs(self.config.VIDEO_OUTPUT_DIR, exist_ok=True)
        
        # Tạo thư mục con theo ngày
        date_str = datetime.now().strftime("%Y%m%d")
        self.daily_dir = os.path.join(self.config.VIDEO_OUTPUT_DIR, date_str)
        os.makedirs(self.daily_dir, exist_ok=True)
    
    def _generate_filename(self, event=False):
        """Tạo filename cho video"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if event:
            prefix = "event"
        else:
            prefix = "continuous"
        
        filename = f"{prefix}_{timestamp}.{self.config.VIDEO_FORMAT}"
        return os.path.join(self.daily_dir, filename)
    
    def _create_writer(self, filename):
        """Tạo video writer với codec phù hợp"""
        fourcc = None
        
        # Chọn codec dựa trên format
        if self.config.VIDEO_FORMAT.lower() in ['mp4', 'avi', 'mov']:
            fourcc = cv2.VideoWriter_fourcc(*self.config.VIDEO_CODEC)
        elif self.config.VIDEO_FORMAT.lower() == 'avi':
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
        else:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        
        # Tạo writer
        writer = cv2.VideoWriter(
            filename,
            fourcc,
            self.config.VIDEO_FPS,
            self.frame_size
        )
        
        if not writer.isOpened():
            print(f"❌ Không thể tạo video writer cho {filename}")
            return None
        
        return writer
    
    def _writer_thread_func(self):
        """Thread function để ghi video"""
        while not self.stop_event.is_set():
            try:
                # Lấy frame từ queue với timeout
                frame_info = self.write_queue.get(timeout=1.0)
                
                if frame_info is None:  # Signal để kết thúc
                    break
                
                frame, timestamp, is_event = frame_info
                
                # Ghi frame
                if self.writer is not None:
                    self.writer.write(frame)
                    self.stats['frames_written'] += 1
                    
                    # Cập nhật metadata
                    self.metadata['frame_count'] += 1
                    
                    # Nếu là event, thêm vào metadata
                    if is_event and self.event_active:
                        if len(self.metadata['events']) == 0 or \
                           self.metadata['events'][-1]['end_time'] is not None:
                            # Bắt đầu event mới
                            event = {
                                'start_time': timestamp,
                                'end_time': None,
                                'frame_offset': self.metadata['frame_count']
                            }
                            self.metadata['events'].append(event)
                            self.stats['events_recorded'] += 1
                
                self.write_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Lỗi trong writer thread: {e}")
                self.stats['frames_dropped'] += 1
    
    def start(self):
        """Bắt đầu ghi video liên tục"""
        if self.recording:
            print("⚠️ Đang ghi video rồi")
            return False
        
        # Tạo filename
        self.video_file = self._generate_filename(event=False)
        
        # Tạo writer
        self.writer = self._create_writer(self.video_file)
        if self.writer is None:
            return False
        
        # Khởi động writer thread
        self.stop_event.clear()
        self.writer_thread = threading.Thread(target=self._writer_thread_func)
        self.writer_thread.daemon = True
        self.writer_thread.start()
        
        # Cập nhật trạng thái
        self.recording = True
        self.metadata['start_time'] = datetime.now().isoformat()
        
        print(f"🎬 Bắt đầu ghi video: {os.path.basename(self.video_file)}")
        return True
    
    def stop(self):
        """Dừng ghi video"""
        if not self.recording:
            return False
        
        # Đánh dấu dừng
        self.recording = False
        
        # Đợi queue trống
        self.write_queue.join()
        
        # Gửi signal để thread kết thúc
        self.write_queue.put(None)
        
        # Đợi thread kết thúc
        if self.writer_thread is not None:
            self.writer_thread.join(timeout=5.0)
        
        # Đóng writer
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        
        # Cập nhật metadata
        self.metadata['duration'] = self.metadata['frame_count'] / self.config.VIDEO_FPS
        
        # Lưu metadata
        self._save_metadata()
        
        print(f"⏹️ Dừng ghi video: {os.path.basename(self.video_file)}")
        print(f"   Frames written: {self.stats['frames_written']}")
        print(f"   Events recorded: {self.stats['events_recorded']}")
        
        return True
    
    def pause(self):
        """Tạm dừng ghi video"""
        if self.recording and not self.paused:
            self.paused = True
            print("⏸️ Tạm dừng ghi video")
            return True
        return False
    
    def resume(self):
        """Tiếp tục ghi video"""
        if self.recording and self.paused:
            self.paused = False
            print("▶️ Tiếp tục ghi video")
            return True
        return False
    
    def start_event_recording(self):
        """Bắt đầu ghi sự kiện"""
        if not self.config.RECORD_EVENTS_ONLY:
            return False
        
        if self.event_recording:
            return True
        
        self.event_recording = True
        self.event_active = True
        self.event_end_counter = 0
        
        # Tạo video file mới cho event
        event_file = self._generate_filename(event=True)
        
        # Đóng writer cũ nếu có
        if self.writer is not None:
            self.writer.release()
        
        # Tạo writer mới
        self.writer = self._create_writer(event_file)
        self.video_file = event_file
        
        # Ghi các frame trong buffer
        for frame_info in list(self.event_buffer):
            self._add_to_queue(frame_info[0], frame_info[1], True)
        
        print(f"🚨 Bắt đầu ghi sự kiện: {os.path.basename(event_file)}")
        return True
    
    def end_event_recording(self):
        """Kết thúc ghi sự kiện"""
        if not self.event_recording:
            return False
        
        self.event_active = False
        self.event_end_counter = self.config.EVENT_POST_BUFFER
        
        # Cập nhật end time cho event cuối cùng
        if self.metadata['events']:
            self.metadata['events'][-1]['end_time'] = datetime.now().isoformat()
        
        print("✅ Kết thúc ghi sự kiện")
        return True
    
    def _add_to_queue(self, frame, timestamp, is_event=False):
        """Thêm frame vào queue để ghi"""
        if not self.recording or self.paused:
            return False
        
        try:
            # Thêm vào buffer nếu đang ở chế độ event
            if self.config.RECORD_EVENTS_ONLY:
                self.event_buffer.append((frame.copy(), timestamp, is_event))
            
            # Thêm vào queue để ghi
            self.write_queue.put_nowait((frame, timestamp, is_event))
            return True
            
        except queue.Full:
            self.stats['frames_dropped'] += 1
            return False
    
    def write(self, frame):
        """Ghi frame vào video"""
        if frame is None or not self.recording or self.paused:
            return False
        
        timestamp = datetime.now().isoformat()
        
        # Kiểm tra xem có đang ở chế độ event không
        if self.config.RECORD_EVENTS_ONLY:
            # Nếu event_active, ghi frame
            if self.event_active:
                success = self._add_to_queue(frame, timestamp, True)
                
                # Nếu đang đếm ngược để kết thúc event
                if self.event_end_counter > 0:
                    self.event_end_counter -= 1
                    if self.event_end_counter == 0:
                        self.event_recording = False
                        if self.writer is not None:
                            self.writer.release()
                            self.writer = None
            else:
                # Chỉ thêm vào buffer
                self.event_buffer.append((frame.copy(), timestamp, False))
        else:
            # Ghi bình thường
            success = self._add_to_queue(frame, timestamp, False)
        
        return success
    
    def trigger_event(self, event_type, data=None):
        """Kích hoạt sự kiện ghi video"""
        if not self.recording:
            return False
        
        print(f"🚨 Kích hoạt sự kiện: {event_type}")
        
        # Bắt đầu ghi sự kiện
        self.start_event_recording()
        
        # Thêm thông tin sự kiện vào metadata
        event_info = {
            'type': event_type,
            'timestamp': datetime.now().isoformat(),
            'data': data or {}
        }
        
        # Lưu event log riêng
        self._log_event(event_info)
        
        return True
    
    def _log_event(self, event_info):
        """Ghi log sự kiện"""
        log_file = os.path.join(self.daily_dir, "events.json")
        
        try:
            events = []
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    events = json.load(f)
            
            events.append(event_info)
            
            with open(log_file, 'w') as f:
                json.dump(events, f, indent=2)
        except Exception as e:
            print(f"❌ Không thể ghi event log: {e}")
    
    def _save_metadata(self):
        """Lưu metadata của video"""
        if self.video_file is None:
            return
        
        metadata_file = self.video_file.replace(
            f".{self.config.VIDEO_FORMAT}", "_metadata.json"
        )
        
        try:
            # Thêm system info
            import psutil
            self.metadata['system_info'] = {
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_percent': psutil.disk_usage('/').percent,
            }
            
            # Thêm recording stats
            self.metadata['recording_stats'] = self.stats
            
            with open(metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            
            print(f"📝 Đã lưu metadata: {os.path.basename(metadata_file)}")
            
        except Exception as e:
            print(f"❌ Không thể lưu metadata: {e}")
    
    def get_status(self):
        """Lấy trạng thái hiện tại"""
        return {
            'recording': self.recording,
            'paused': self.paused,
            'event_recording': self.event_recording,
            'event_active': self.event_active,
            'video_file': os.path.basename(self.video_file) if self.video_file else None,
            'frame_count': self.metadata['frame_count'],
            'queue_size': self.write_queue.qsize(),
            'buffer_size': len(self.event_buffer),
            'stats': self.stats.copy()
        }
    
    def release(self):
        """Giải phóng tất cả resource"""
        # Dừng recording nếu đang chạy
        if self.recording:
            self.stop()
        
        # Đảm bảo thread đã dừng
        self.stop_event.set()
        
        # Đóng writer
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        
        print("🗑️ Recorder đã được giải phóng")