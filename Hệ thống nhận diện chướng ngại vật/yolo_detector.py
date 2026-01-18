import cv2
import numpy as np
from ultralytics import YOLO

class YOLOObstacleDetector:
    """
    YOLO v8 Obstacle Detector - Cải tiến
    Phát hiện người, xe (các loại xe) từ video hoặc camera
    Xác định chính xác với confidence cao hơn
    """
    
    def __init__(self, model_name="yolov8n.pt"):
        """
        Khởi tạo YOLO detector
        
        Args:
            model_name (str): Tên model YOLOv8 (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
        """
        print(f"[INFO] Loading YOLOv8 model: {model_name}...")
        self.model = YOLO(model_name)
        self.model.to('cpu')  # Dùng CPU (có GPU thì tự switch)
        print(f"[OK] Model loaded successfully!")
        
        # Class IDs cần phát hiện (từ COCO dataset)
        # Thêm các class khác để bắt tốt hơn
        self.target_classes = {
            0: "Nguoi",          # person
            1: "Xe dap",         # bicycle
            2: "Xe o to",        # car
            3: "Xe may",         # motorcycle
            4: "Xe phuong tien",  # airplane
            5: "Bus",            # bus
            6: "Tau",            # train
            7: "Xe tai",         # truck
            9: "Van tai",        # boat (water transportation)
        }
        
        # Màu cho mỗi class (BGR format cho OpenCV)
        self.colors = {
            0: (0, 255, 255),    # Người - Yellow
            1: (255, 255, 0),    # Xe đạp - Cyan
            2: (0, 0, 255),      # Xe ô tô - Red
            3: (255, 0, 0),      # Xe máy - Blue
            4: (0, 165, 255),    # Máy bay - Orange
            5: (0, 255, 0),      # Bus - Green
            6: (128, 0, 128),    # Tàu - Purple
            7: (255, 0, 255),    # Xe tải - Magenta
            9: (0, 128, 255),    # Vận tải - Dark Orange
        }
        
        # Confidence threshold tối thiểu cho từng loại
        self.class_thresholds = {
            0: 0.3,   # Người
            1: 0.3,   # Xe đạp
            2: 0.3,   # Xe ô tô
            3: 0.3,   # Xe máy
            4: 0.4,   # Máy bay
            5: 0.3,   # Bus
            6: 0.3,   # Tàu
            7: 0.3,   # Xe tải
            9: 0.35,  # Vận tải
        }

    def detect(self, frame, conf=0.4):
        """
        Phát hiện chướng ngại vật trong frame - CẢI TIẾN
        
        Args:
            frame (np.ndarray): Hình ảnh input (BGR format từ OpenCV)
            conf (float): Confidence threshold tổng (0-1, mặc định 0.4)
            
        Returns:
            tuple: (frame_with_boxes, detections_list)
                - frame_with_boxes: Frame có vẽ bounding box + nhãn rõ ràng
                - detections_list: Danh sách các detection
        """
        try:
            # Thực hiện detection
            results = self.model(frame, conf=conf, verbose=False)
            
            frame_copy = frame.copy()
            detections_list = []
            
            # Xử lý kết quả detection
            if results and len(results) > 0:
                result = results[0]
                
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes
                    
                    for box in boxes:
                        # Lấy thông tin detection
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        
                        # Chỉ xử lý các class mục tiêu
                        if class_id in self.target_classes:
                            # Kiểm tra threshold cụ thể cho class
                            class_threshold = self.class_thresholds.get(class_id, 0.35)
                            if confidence < class_threshold:
                                continue
                            
                            # Lấy tọa độ bounding box
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            
                            # Lấy tên class và màu
                            class_name = self.target_classes[class_id]
                            color = self.colors.get(class_id, (255, 255, 255))
                            
                            # ===== VẼ BOUNDING BOX (dày hơn, nổi bật hơn) =====
                            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 4)
                            
                            # ===== VẼ NHÃN RỰ (dùng font TO HƠN) =====
                            label = f"{class_name}: {confidence:.1%}"
                            font = cv2.FONT_HERSHEY_SIMPLEX  # Font đúng (không có BOLD)
                            font_scale = 0.8  # Bé lại một chút
                            thickness = 2    # Bé lại
                            
                            label_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
                            
                            # Background cho text (nhỏ hơn)
                            bg_pad = 5  # Nhỏ lại
                            cv2.rectangle(frame_copy, 
                                        (x1 - bg_pad, y1 - label_size[1] - bg_pad),
                                        (x1 + label_size[0] + bg_pad, y1 + bg_pad), 
                                        color, -1)
                            
                            # Viền ngoài cho background (nổi bật)
                            cv2.rectangle(frame_copy, 
                                        (x1 - bg_pad, y1 - label_size[1] - bg_pad),
                                        (x1 + label_size[0] + bg_pad, y1 + bg_pad), 
                                        (0, 0, 0), 2)
                            
                            # Vẽ text (màu trắng nổi bật)
                            cv2.putText(frame_copy, label, 
                                      (x1 + bg_pad // 2, y1 - bg_pad // 2),
                                      font, font_scale, (255, 255, 255), thickness)
                            
                            # Thêm vào danh sách detections
                            detections_list.append({
                                'class_id': class_id,
                                'class_name': class_name,
                                'confidence': confidence,
                                'bbox': (x1, y1, x2, y2),
                                'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                                'width': x2 - x1,
                                'height': y2 - y1,
                                'area': (x2 - x1) * (y2 - y1)
                            })
            
            return frame_copy, detections_list
            
        except Exception as e:
            print(f"[ERROR] Detection error: {str(e)}")
            return frame, []

    def get_target_classes(self):
        """Trả về danh sách các class mục tiêu"""
        return self.target_classes

    def get_detections_summary(self, detections_list):
        """
        Tóm tắt số lượng detections theo loại
        
        Args:
            detections_list (list): Danh sách detections từ phương thức detect()
            
        Returns:
            dict: Tóm tắt theo class
        """
        summary = {}
        for detection in detections_list:
            class_name = detection['class_name']
            summary[class_name] = summary.get(class_name, 0) + 1
        return summary

    def get_detection_stats(self, detections_list):
        """
        Lấy thống kê chi tiết về detections
        
        Args:
            detections_list (list): Danh sách detections
            
        Returns:
            dict: Thống kê (tổng, theo class, confidence trung bình)
        """
        if not detections_list:
            return {
                'total': 0,
                'by_class': {},
                'avg_confidence': 0
            }
        
        summary = self.get_detections_summary(detections_list)
        total_confidence = sum(det['confidence'] for det in detections_list)
        
        return {
            'total': len(detections_list),
            'by_class': summary,
            'avg_confidence': total_confidence / len(detections_list)
        }

