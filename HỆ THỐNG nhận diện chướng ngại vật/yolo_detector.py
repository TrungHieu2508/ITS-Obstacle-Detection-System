import cv2
import numpy as np
from ultralytics import YOLO


class YOLOObstacleDetector:
    """
    YOLOv8 Obstacle Detector (Phiên bản cải tiến – chỉ clean code)

    Chức năng:
    - Phát hiện người và các loại phương tiện từ video / camera
    - Áp dụng confidence threshold riêng cho từng class
    - Vẽ bounding box + nhãn rõ ràng, dễ quan sát
    - Trả về danh sách detection chi tiết để xử lý tiếp
    """

    def __init__(self, model_name="yolov8n.pt"):
        """
        Khởi tạo YOLO detector

        Args:
            model_name (str):
                Tên model YOLOv8
                (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
        """

        # ===== LOAD MODEL =====
        print(f"[INFO] Loading YOLOv8 model: {model_name}...")
        self.model = YOLO(model_name)

        # Sử dụng CPU (nếu có GPU, Ultralytics sẽ tự động tối ưu)
        self.model.to("cpu")
        print("[OK] Model loaded successfully!")

        # ===== CLASS MỤC TIÊU (COCO DATASET) =====
        # Key  : class_id theo COCO
        # Value: tên hiển thị (Việt hóa)
        self.target_classes = {
            0: "Nguoi",           # person
            1: "Xe dap",          # bicycle
            2: "Xe o to",         # car
            3: "Xe may",          # motorcycle
            4: "Xe phuong tien",  # airplane
            5: "Bus",             # bus
            6: "Tau",             # train
            7: "Xe tai",          # truck
            9: "Van tai",         # boat
        }

        # ===== ROI (REGION OF INTEREST) =====
        # Hiện tại CHƯA sử dụng trong detect()
        # Dùng để mở rộng sau: cảnh báo vùng nguy hiểm
        self.roi = None
        self.roi_type = "rectangle"  # "rectangle" hoặc "polygon"

        # ===== MÀU SẮC THEO CLASS (BGR – OpenCV) =====
        self.colors = {
            0: (0, 255, 255),    # Người – Vàng
            1: (255, 255, 0),    # Xe đạp – Cyan
            2: (0, 75, 150),     # Xe ô tô – Nâu
            3: (255, 0, 0),      # Xe máy – Xanh dương
            4: (0, 165, 255),    # Máy bay – Cam
            5: (0, 255, 0),      # Bus – Xanh lá
            6: (128, 0, 128),    # Tàu – Tím
            7: (255, 0, 255),    # Xe tải – Hồng
            9: (0, 128, 255),    # Vận tải – Cam đậm
        }

        # ===== CONFIDENCE THRESHOLD RIÊNG CHO TỪNG CLASS =====
        self.class_thresholds = {
            0: 0.30,   # Người
            1: 0.30,   # Xe đạp
            2: 0.30,   # Xe ô tô
            3: 0.30,   # Xe máy
            4: 0.40,   # Máy bay
            5: 0.30,   # Bus
            6: 0.30,   # Tàu
            7: 0.30,   # Xe tải
            9: 0.35,   # Vận tải
        }

        # ===== TRACKING (CHƯA ÁP DỤNG TRONG detect()) =====
        # Để mở rộng rule: vật thể tồn tại >= 3 giây
        self.tracked_objects = {}
        self.object_id_counter = 0
        self.time_threshold_seconds = 3
        self.iou_threshold = 0.3

    def detect(self, frame, conf=0.4):
        """
        Phát hiện chướng ngại vật trong một frame

        Args:
            frame (np.ndarray):
                Ảnh đầu vào (BGR – OpenCV)
            conf (float):
                Confidence threshold tổng cho YOLO (mặc định 0.4)

        Returns:
            tuple:
                frame_with_boxes (np.ndarray):
                    Frame đã vẽ bounding box + label
                detections_list (list):
                    Danh sách detection chi tiết
        """

        try:
            # ===== YOLO INFERENCE =====
            results = self.model(frame, conf=conf, verbose=False)

            frame_copy = frame.copy()
            detections_list = []

            # ===== KIỂM TRA KẾT QUẢ =====
            if results and len(results) > 0:
                result = results[0]

                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes

                    for box in boxes:
                        # Lấy class ID và confidence
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])

                        # Chỉ xử lý class nằm trong target_classes
                        if class_id not in self.target_classes:
                            continue

                        # Kiểm tra threshold riêng cho từng class
                        class_threshold = self.class_thresholds.get(class_id, 0.35)
                        if confidence < class_threshold:
                            continue

                        # Lấy tọa độ bounding box
                        x1, y1, x2, y2 = map(int, box.xyxy[0])

                        # Thông tin hiển thị
                        class_name = self.target_classes[class_id]
                        color = self.colors.get(class_id, (255, 255, 255))

                        # ===== VẼ BOUNDING BOX =====
                        cv2.rectangle(
                            frame_copy,
                            (x1, y1),
                            (x2, y2),
                            color,
                            4
                        )

                        # ===== VẼ NHÃN =====
                        label = f"{class_name}: {confidence:.1%}"
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.8
                        thickness = 2

                        label_size = cv2.getTextSize(
                            label, font, font_scale, thickness
                        )[0]

                        bg_pad = 5

                        # Nền label
                        cv2.rectangle(
                            frame_copy,
                            (x1 - bg_pad, y1 - label_size[1] - bg_pad),
                            (x1 + label_size[0] + bg_pad, y1 + bg_pad),
                            color,
                            -1
                        )

                        # Viền nền label
                        cv2.rectangle(
                            frame_copy,
                            (x1 - bg_pad, y1 - label_size[1] - bg_pad),
                            (x1 + label_size[0] + bg_pad, y1 + bg_pad),
                            (0, 0, 0),
                            2
                        )

                        # Text label
                        cv2.putText(
                            frame_copy,
                            label,
                            (x1 + bg_pad // 2, y1 - bg_pad // 2),
                            font,
                            font_scale,
                            (255, 255, 255),
                            thickness
                        )

                        # ===== LƯU DETECTION =====
                        detections_list.append({
                            "class_id": class_id,
                            "class_name": class_name,
                            "confidence": confidence,
                            "bbox": (x1, y1, x2, y2),
                            "center": ((x1 + x2) // 2, (y1 + y2) // 2),
                            "width": x2 - x1,
                            "height": y2 - y1,
                            "area": (x2 - x1) * (y2 - y1),
                        })

            return frame_copy, detections_list

        except Exception as e:
            print(f"[ERROR] Detection error: {str(e)}")
            return frame, []

    def get_target_classes(self):
        """
        Trả về danh sách class mục tiêu
        """
        return self.target_classes

    def get_detections_summary(self, detections_list):
        """
        Tóm tắt số lượng detection theo từng class

        Args:
            detections_list (list):
                Danh sách detection

        Returns:
            dict:
                {class_name: count}
        """
        summary = {}
        for det in detections_list:
            name = det["class_name"]
            summary[name] = summary.get(name, 0) + 1
        return summary

    def get_detection_stats(self, detections_list):
        """
        Thống kê chi tiết detection

        Args:
            detections_list (list):
                Danh sách detection

        Returns:
            dict:
                - total: tổng số object
                - by_class: số lượng theo class
                - avg_confidence: confidence trung bình
        """

        if not detections_list:
            return {
                "total": 0,
                "by_class": {},
                "avg_confidence": 0,
            }

        summary = self.get_detections_summary(detections_list)
        total_confidence = sum(det["confidence"] for det in detections_list)

        return {
            "total": len(detections_list),
            "by_class": summary,
            "avg_confidence": total_confidence / len(detections_list),
        }


    # ===================== DISTANCE CALCULATION & NEAREST OBSTACLE =====================

    def calculate_proximity_score(self, detection, frame_height):
        """
        Tính độ gần (proximity score) của một vật thể.

        Nguyên lý:
        - Bounding box càng lớn  → vật thể càng gần camera
        - Vật thể càng nằm thấp (gần đáy frame) → càng gần

        Công thức:
        proximity_score = bbox_area + (y2 * 0.5)

        Args:
            detection (dict):
                Thông tin detection (bbox, class, confidence, ...)
            frame_height (int):
                Chiều cao của frame (pixel)

        Returns:
            float:
                Proximity score (giá trị càng cao → vật thể càng gần)
        """

        # Lấy bounding box
        x1, y1, x2, y2 = detection["bbox"]

        # ===== DIỆN TÍCH BOUNDING BOX =====
        bbox_area = (x2 - x1) * (y2 - y1)

        # ===== VỊ TRÍ THEO CHIỀU DỌC =====
        # y2 càng lớn → càng gần đáy frame → càng gần camera
        distance_from_bottom = frame_height - y2

        # ===== TÍNH PROXIMITY SCORE =====
        # - bbox_area: đại diện độ lớn vật thể
        # - (frame_height - distance_from_bottom) == y2
        proximity_score = bbox_area + (frame_height - distance_from_bottom) * 0.5

        return proximity_score

    def get_nearest_obstacle(self, detections_list, frame_height):
        """
        Tìm vật thể gần nhất trong danh sách detections.

        Tiêu chí:
        - So sánh proximity_score
        - Score cao nhất → vật thể gần nhất

        Args:
            detections_list (list):
                Danh sách detection từ detect()
            frame_height (int):
                Chiều cao frame

        Returns:
            dict | None:
                Detection của vật thể gần nhất (kèm proximity_score)
                None nếu không có detection
        """

        if not detections_list:
            return None

        nearest = None
        max_score = 0

        for detection in detections_list:
            score = self.calculate_proximity_score(detection, frame_height)

            if score > max_score:
                max_score = score
                nearest = detection

        # Gắn thêm proximity_score vào detection
        if nearest is not None:
            nearest["proximity_score"] = max_score

        return nearest

    def get_nearest_obstacle_with_tracking_id(self, frame_height):
        """
        Tìm vật thể gần nhất từ tracked_objects.

        Mục đích:
        - Dùng cho logic cảnh báo nâng cao
        - Giữ đúng object_id (không bị nhảy giữa các frame)

        Args:
            frame_height (int):
                Chiều cao frame

        Returns:
            dict | None:
                {
                    object_id,
                    class_name,
                    last_detection,
                    proximity_score,
                    bbox
                }
                hoặc None nếu không có tracked object
        """

        if not self.tracked_objects:
            return None

        nearest = None
        max_score = 0

        for object_id, obj_info in self.tracked_objects.items():

            # Chỉ xử lý object có detection gần nhất
            if "last_detection" not in obj_info:
                continue

            detection = obj_info["last_detection"]
            score = self.calculate_proximity_score(detection, frame_height)

            if score > max_score:
                max_score = score
                nearest = {
                    "object_id": object_id,
                    "class_name": obj_info["class_name"],
                    "last_detection": detection,
                    "proximity_score": score,
                    "bbox": detection["bbox"],
                }

        return nearest

    def draw_nearest_indicator(self, frame, nearest_detection):
        """
        Vẽ indicator cho vật thể gần nhất.

        Hiển thị:
        - Vòng tròn lớn bao quanh bbox
        - Mũi tên chỉ lên trên
        - Nhãn "NEAREST"

        Args:
            frame (np.ndarray):
                Frame gốc
            nearest_detection (dict):
                Detection của vật thể gần nhất

        Returns:
            np.ndarray:
                Frame đã được vẽ indicator
        """

        if not nearest_detection:
            return frame

        frame_copy = frame.copy()
        x1, y1, x2, y2 = nearest_detection["bbox"]

        # ===== TÂM BOUNDING BOX =====
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # ===== VẼ VÒNG TRÒN NỔI BẬT =====
        radius = max(x2 - x1, y2 - y1) // 2 + 20

        cv2.circle(frame_copy, (center_x, center_y), radius, (0, 255, 255), 3)
        cv2.circle(frame_copy, (center_x, center_y), radius + 5, (0, 255, 255), 1)

        # ===== VẼ MŨI TÊN =====
        cv2.arrowedLine(
            frame_copy,
            (center_x, center_y - radius),
            (center_x, center_y - radius - 25),
            (0, 255, 255),
            3,
            tipLength=0.3,
        )

        # ===== TEXT "NEAREST" =====
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = " NEAREST"
        text_size = cv2.getTextSize(text, font, 1.2, 2)[0]

        text_x = center_x - text_size[0] // 2
        text_y = center_y - radius - 60

        # Background text
        cv2.rectangle(
            frame_copy,
            (text_x - 5, text_y - text_size[1] - 5),
            (text_x + text_size[0] + 5, text_y + 5),
            (0, 255, 255),
            -1,
        )

        cv2.putText(
            frame_copy,
            text,
            (text_x, text_y),
            font,
            1.2,
            (0, 0, 0),
            2,
        )

        return frame_copy

    def draw_red_flash(self, frame, flash_intensity=0.6):
        """
        Vẽ hiệu ứng flash đỏ toàn màn hình (cảnh báo nguy hiểm).

        Args:
            frame (np.ndarray):
                Frame gốc
            flash_intensity (float):
                Cường độ flash (0 → 1)
                0.6 = đỏ rõ nhưng vẫn nhìn được nội dung

        Returns:
            np.ndarray:
                Frame đã phủ flash đỏ
        """

        frame_copy = frame.copy()

        # ===== TẠO OVERLAY MÀU ĐỎ =====
        overlay = np.zeros_like(frame_copy)
        overlay[:, :] = (0, 0, 255)  # BGR: Red

        # ===== BLEND OVERLAY =====
        frame_copy = cv2.addWeighted(
            overlay,
            flash_intensity,
            frame_copy,
            1 - flash_intensity,
            0,
        )

        return frame_copy


    # ===================== HÀM XỬ LÝ ROI =====================

    def set_roi_rectangle(self, x1, y1, x2, y2):
        """
        Định nghĩa vùng nguy hiểm (ROI) dạng hình chữ nhật
        
        Args:
            x1, y1, x2, y2 (int): Tọa độ góc trên trái (x1, y1) và góc dưới phải (x2, y2)
        """
        self.roi = [x1, y1, x2, y2]
        self.roi_type = "rectangle"
        print(f"[INFO] ROI set: ({x1}, {y1}) -> ({x2}, {y2})")

    def set_roi_polygon(self, points):
        """
        Định nghĩa vùng nguy hiểm (ROI) dạng đa giác
        
        Args:
            points (list): Danh sách các điểm [(x1, y1), (x2, y2), ...]
        """
        self.roi = np.array(points, dtype=np.int32)
        self.roi_type = "polygon"
        print(f"[INFO] ROI polygon set with {len(points)} points")

    def is_point_in_roi(self, point):
        """
        Kiểm tra một điểm có nằm trong ROI hay không
        
        Args:
            point (tuple): Tọa độ điểm (x, y)
            
        Returns:
            bool: True nếu điểm trong ROI, False nếu không
        """
        if self.roi is None:
            return False
        
        if self.roi_type == "rectangle":
            x, y = point
            x1, y1, x2, y2 = self.roi
            return x1 <= x <= x2 and y1 <= y <= y2
        
        elif self.roi_type == "polygon":
            # Dùng hàm pointPolygonTest của OpenCV
            return cv2.pointPolygonTest(self.roi, point, False) >= 0
        
        return False

    def is_bbox_in_roi(self, bbox):
        """
        Kiểm tra bounding box có nằm trong ROI hay không
        Trả về phần trăm diện tích bbox overlap với ROI
        
        Args:
            bbox (tuple): (x1, y1, x2, y2)
            
        Returns:
            float: Phần trăm overlap (0-100), 0 nếu không overlap
        """
        if self.roi is None:
            return 0
        
        x1_bbox, y1_bbox, x2_bbox, y2_bbox = bbox
        bbox_area = (x2_bbox - x1_bbox) * (y2_bbox - y1_bbox)
        
        if bbox_area == 0:
            return 0
        
        if self.roi_type == "rectangle":
            x1_roi, y1_roi, x2_roi, y2_roi = self.roi
            
            # Tính diện tích overlap
            x1_overlap = max(x1_bbox, x1_roi)
            y1_overlap = max(y1_bbox, y1_roi)
            x2_overlap = min(x2_bbox, x2_roi)
            y2_overlap = min(y2_bbox, y2_roi)
            
            if x1_overlap < x2_overlap and y1_overlap < y2_overlap:
                overlap_area = (x2_overlap - x1_overlap) * (y2_overlap - y1_overlap)
                return (overlap_area / bbox_area) * 100
        
        elif self.roi_type == "polygon":
            # Kiểm tra tâm bbox có trong polygon
            center_x = (x1_bbox + x2_bbox) // 2
            center_y = (y1_bbox + y2_bbox) // 2
            if self.is_point_in_roi((center_x, center_y)):
                return 100
        
        return 0

    def get_detections_in_roi(self, detections_list):
        """
        Lọc các detections nằm trong ROI
        
        Args:
            detections_list (list): Danh sách detections
            
        Returns:
            list: Danh sách detections có trong ROI (thêm trường 'roi_overlap')
        """
        detections_in_roi = []
        
        for detection in detections_list:
            overlap = self.is_bbox_in_roi(detection['bbox'])
            if overlap > 0:
                detection['roi_overlap'] = overlap
                detections_in_roi.append(detection)
        
        return detections_in_roi

    def draw_roi(self, frame, color=(0, 255, 0), thickness=3, alpha=0.3):
        """
        Vẽ ROI lên frame
        
        Args:
            frame (np.ndarray): Frame input
            color (tuple): Màu RGB (BGR cho OpenCV)
            thickness (int): Độ dày đường
            alpha (float): Độ trong suốt khi fill (0-1)
            
        Returns:
            np.ndarray: Frame đã vẽ ROI
        """
        if self.roi is None:
            return frame
        
        frame_copy = frame.copy()
        overlay = frame.copy()
        
        if self.roi_type == "rectangle":
            x1, y1, x2, y2 = self.roi
            # Vẽ rectangle
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            # Vẽ viền
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, thickness)
        
        elif self.roi_type == "polygon":
            # Vẽ polygon
            cv2.fillPoly(overlay, [self.roi], color)
            cv2.polylines(frame_copy, [self.roi], True, color, thickness)
        
        # Blend overlay với frame gốc
        frame_copy = cv2.addWeighted(overlay, alpha, frame_copy, 1 - alpha, 0)
        cv2.addWeighted(frame_copy, 1, frame_copy, 0, 0, frame_copy)
        
        return frame_copy

    def draw_alert(self, frame, detections_in_roi, alert_message="⚠️ CHƯỚNG NGẠI VẬT PHÁT HIỆN!"):
        """
        Vẽ cảnh báo lớn lên frame nếu phát hiện vật cản trong ROI
        
        Args:
            frame (np.ndarray): Frame input
            detections_in_roi (list): Danh sách detections trong ROI
            alert_message (str): Thông báo cảnh báo
            
        Returns:
            np.ndarray: Frame đã vẽ cảnh báo
        """
        if not detections_in_roi:
            return frame
        
        frame_copy = frame.copy()
        h, w = frame.shape[:2]
        
        # Vẽ nền đỏ nhấp nháy ở đầu frame
        alert_height = 80
        cv2.rectangle(frame_copy, (0, 0), (w, alert_height), (0, 0, 255), -1)
        
        # Viền vàng để nổi bật
        cv2.rectangle(frame_copy, (0, 0), (w, alert_height), (0, 255, 255), 4)
        
        # Đếm số vật cản
        object_count = len(detections_in_roi)
        
        # Hiển thị thông báo chính
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        thickness = 3
        
        alert_text = f"{alert_message} ({object_count} object)"
        text_size = cv2.getTextSize(alert_text, font, font_scale, thickness)[0]
        x_text = (w - text_size[0]) // 2
        y_text = 50
        
        cv2.putText(frame_copy, alert_text, (x_text, y_text), 
                   font, font_scale, (255, 255, 255), thickness)
        
        # Hiển thị danh sách các vật cản phát hiện
        detail_text = " | ".join([f"{d['class_name']}: {d['confidence']:.0%}" 
                                  for d in detections_in_roi[:3]])
        if len(detections_in_roi) > 3:
            detail_text += f" +{len(detections_in_roi) - 3} more"
        
        font_scale_small = 0.6
        thickness_small = 1
        text_size_small = cv2.getTextSize(detail_text, font, font_scale_small, thickness_small)[0]
        x_detail = (w - text_size_small[0]) // 2
        
        cv2.putText(frame_copy, detail_text, (x_detail, 70),
                   font, font_scale_small, (255, 255, 255), thickness_small)
        
        return frame_copy

    def detect_with_roi(self, frame, conf=0.4, show_roi=True, show_alert=True):
        """
        Phát hiện chướng ngại vật + kiểm tra ROI + cảnh báo
        
        Args:
            frame (np.ndarray): Frame input
            conf (float): Confidence threshold
            show_roi (bool): Có vẽ ROI hay không
            show_alert (bool): Có hiển thị cảnh báo hay không
            
        Returns:
            tuple: (frame_result, detections_list, detections_in_roi)
        """
        # Phát hiện chướng ngại vật
        frame_detected, detections = self.detect(frame, conf=conf)
        
        # Kiểm tra vật cản trong ROI
        detections_in_roi = self.get_detections_in_roi(detections)
        
        # Vẽ ROI
        if show_roi and self.roi is not None:
            frame_detected = self.draw_roi(frame_detected)
        
        # Vẽ cảnh báo nếu có vật cản trong ROI
        if show_alert and detections_in_roi:
            frame_detected = self.draw_alert(frame_detected, detections_in_roi)
        
        return frame_detected, detections, detections_in_roi

    # ===================== OBJECT TRACKING & 3-SECOND RULE =====================

    def calculate_iou(self, bbox1, bbox2):
        """
        Tính Intersection over Union (IoU) giữa hai bounding box
        
        Args:
            bbox1, bbox2 (tuple): (x1, y1, x2, y2)
            
        Returns:
            float: IoU value (0-1)
        """
        x1_bbox1, y1_bbox1, x2_bbox1, y2_bbox1 = bbox1
        x1_bbox2, y1_bbox2, x2_bbox2, y2_bbox2 = bbox2
        
        # Tính giao diện
        x1_inter = max(x1_bbox1, x1_bbox2)
        y1_inter = max(y1_bbox1, y1_bbox2)
        x2_inter = min(x2_bbox1, x2_bbox2)
        y2_inter = min(y2_bbox1, y2_bbox2)
        
        if x1_inter >= x2_inter or y1_inter >= y2_inter:
            return 0.0
        
        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        
        # Tính diện tích của từng box
        area1 = (x2_bbox1 - x1_bbox1) * (y2_bbox1 - y1_bbox1)
        area2 = (x2_bbox2 - x1_bbox2) * (y2_bbox2 - y1_bbox2)
        
        union_area = area1 + area2 - inter_area
        
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area

    def match_detection_to_tracked(self, detection):
        """
        Ghép (match) detection hiện tại với tracked object cũ dựa trên IoU
        
        Args:
            detection (dict): Detection hiện tại
            
        Returns:
            int: ID của tracked object nếu có match, -1 nếu không
        """
        best_match_id = -1
        best_iou = self.iou_threshold
        
        for obj_id, obj_info in self.tracked_objects.items():
            iou = self.calculate_iou(detection['bbox'], obj_info['last_bbox'])
            if iou > best_iou:
                best_iou = iou
                best_match_id = obj_id
        
        return best_match_id

    def update_tracking(self, detections, frame_number, fps=30):
        """
        Cập nhật tracking cho detections hiện tại
        Quy tắc 3 giây: Nếu vật thể xuất hiện từ 0-2.9s, vẫn đếm thời gian
                       Nếu từ 3.1s trở đi mà vẫn còn trong ROI -> VẬT CẢN
        
        Args:
            detections (list): Danh sách detections hiện tại
            frame_number (int): Số frame hiện tại (bắt đầu từ 0)
            fps (int): Frame per second của video
            
        Returns:
            dict: Info về các vật cản nguy hiểm
                {
                    'dangerous_objects': [...],  # Các vật thể vượt quá 3s
                    'tracked_objects': {...},    # Tất cả tracked objects
                    'time_elapsed': {...}        # Thời gian elapsed cho từng object
                }
        """
        matched_ids = set()
        time_per_frame = 1.0 / fps
        
        # Match detections với tracked objects
        for detection in detections:
            matched_id = self.match_detection_to_tracked(detection)
            
            if matched_id != -1:
                # Cập nhật object đã tracked
                matched_ids.add(matched_id)
                self.tracked_objects[matched_id]['last_bbox'] = detection['bbox']
                self.tracked_objects[matched_id]['last_detection'] = detection
                self.tracked_objects[matched_id]['frame_count'] += 1
                self.tracked_objects[matched_id]['last_frame_number'] = frame_number
            else:
                # Vật thể mới - tạo tracking mới
                self.object_id_counter += 1
                new_id = self.object_id_counter
                self.tracked_objects[new_id] = {
                    'object_id': new_id,
                    'first_frame_number': frame_number,
                    'last_frame_number': frame_number,
                    'frame_count': 1,
                    'first_detection': detection,
                    'last_detection': detection,
                    'last_bbox': detection['bbox'],
                    'class_name': detection['class_name'],
                    'class_id': detection['class_id']
                }
                matched_ids.add(new_id)
        
        # Loại bỏ tracked objects không được detect trong frame này
        # (ngoài timeout window)
        timeout_frames = int(self.time_threshold_seconds * fps * 0.5)  # 50% từ 3s
        objects_to_remove = []
        for obj_id, obj_info in self.tracked_objects.items():
            if obj_id not in matched_ids:
                if frame_number - obj_info['last_frame_number'] > timeout_frames:
                    objects_to_remove.append(obj_id)
        
        for obj_id in objects_to_remove:
            del self.tracked_objects[obj_id]
        
        # Tính toán dangerous objects (vượt quá 3 giây)
        dangerous_objects = []
        time_elapsed_dict = {}
        
        for obj_id, obj_info in self.tracked_objects.items():
            frames_existed = obj_info['last_frame_number'] - obj_info['first_frame_number'] + 1
            time_elapsed = frames_existed * time_per_frame
            time_elapsed_dict[obj_id] = time_elapsed
            
            # Nếu vượt quá 3 giây -> vật cản nguy hiểm
            if time_elapsed > self.time_threshold_seconds:
                dangerous_objects.append({
                    'object_id': obj_id,
                    'class_name': obj_info['class_name'],
                    'time_elapsed': time_elapsed,
                    'frame_count': obj_info['frame_count'],
                    'last_detection': obj_info['last_detection']
                })
        
        return {
            'dangerous_objects': dangerous_objects,
            'tracked_objects': self.tracked_objects,
            'time_elapsed': time_elapsed_dict,
            'total_tracked': len(self.tracked_objects)
        }

    def reset_tracking(self):
        """
        Reset toàn bộ tracking data (gọi khi bắt đầu video mới)
        """
        self.tracked_objects = {}
        self.object_id_counter = 0
        print("[INFO] Tracking data reset")

    def draw_dangerous_obstacles(self, frame, tracking_result):
        """
        Vẽ bounding box đỏ lớn xung quanh các vật thể nguy hiểm (>3s)
        để dễ nhận biết chúng là chướng ngại vật
        
        Args:
            frame (np.ndarray): Frame input
            tracking_result (dict): Kết quả từ update_tracking()
            
        Returns:
            np.ndarray: Frame đã vẽ bounding box đỏ cho vật thể nguy hiểm
        """
        if not tracking_result or not tracking_result['dangerous_objects']:
            return frame
        
        frame_copy = frame.copy()
        RED = (0, 0, 255)  # Màu đỏ (BGR format)
        THICK = 6  # Độ dày viền to hơn để nổi bật
        
        dangerous_objects = tracking_result['dangerous_objects']
        
        for danger_obj in dangerous_objects:
            # Lấy bbox từ last_detection
            if 'last_detection' in danger_obj:
                bbox = danger_obj['last_detection']['bbox']
                x1, y1, x2, y2 = bbox
                
                # Vẽ bounding box đỏ dày
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), RED, THICK)
                
                # Vẽ góc (4 góc vuông nổi bật)
                corner_length = 30
                corner_thick = 4
                
                # Góc trên trái
                cv2.line(frame_copy, (x1, y1), (x1 + corner_length, y1), RED, corner_thick)
                cv2.line(frame_copy, (x1, y1), (x1, y1 + corner_length), RED, corner_thick)
                
                # Góc trên phải
                cv2.line(frame_copy, (x2, y1), (x2 - corner_length, y1), RED, corner_thick)
                cv2.line(frame_copy, (x2, y1), (x2, y1 + corner_length), RED, corner_thick)
                
                # Góc dưới trái
                cv2.line(frame_copy, (x1, y2), (x1 + corner_length, y2), RED, corner_thick)
                cv2.line(frame_copy, (x1, y2), (x1, y2 - corner_length), RED, corner_thick)
                
                # Góc dưới phải
                cv2.line(frame_copy, (x2, y2), (x2 - corner_length, y2), RED, corner_thick)
                cv2.line(frame_copy, (x2, y2), (x2, y2 - corner_length), RED, corner_thick)
                
                # Vẽ label "DANGER" trên bbox
                time_elapsed = danger_obj['time_elapsed']
                label = f" DANGER: {time_elapsed:.1f}s"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.0
                thickness = 2
                
                label_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
                
                # Background label (đỏ thẫm)
                label_y = max(y1 - 10, label_size[1] + 5)
                cv2.rectangle(frame_copy, 
                            (x1, label_y - label_size[1] - 8),
                            (x1 + label_size[0] + 10, label_y + 5),
                            RED, -1)
                
                # Viền trắng cho label
                cv2.rectangle(frame_copy,
                            (x1, label_y - label_size[1] - 8),
                            (x1 + label_size[0] + 10, label_y + 5),
                            (255, 255, 255), 2)
                
                # Text màu vàng
                cv2.putText(frame_copy, label,
                          (x1 + 5, label_y - 3),
                          font, font_scale, (0, 255, 255), thickness)
        
        return frame_copy

    def draw_tracking_info(self, frame, tracking_result):
        """
        Vẽ thông tin tracking lên frame
        
        Args:
            frame (np.ndarray): Frame input
            tracking_result (dict): Kết quả từ update_tracking()
            
        Returns:
            np.ndarray: Frame đã vẽ thông tin tracking
        """
        frame_copy = frame.copy()
        h, w = frame.shape[:2]
        
        dangerous_objects = tracking_result['dangerous_objects']
        time_elapsed = tracking_result['time_elapsed']
        
        # Vẽ thông tin tracked objects lên frame
        y_offset = 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        
        for obj_id in sorted(time_elapsed.keys()):
            elapsed = time_elapsed[obj_id]
            obj_info = self.tracked_objects[obj_id]
            
            # Màu khác nhau tùy theo có phải vật cản hay không
            if elapsed > self.time_threshold_seconds:
                color = (0, 0, 255)  # Đỏ - Nguy hiểm
                status = "⚠️ DANGEROUS"
            else:
                color = (0, 255, 0)  # Xanh - Bình thường
                status = f"TRACKING ({elapsed:.1f}s)"
            
        #     text = f"ID:{obj_id} {obj_info['class_name']} | {status}"
        #     cv2.putText(frame_copy, text, (10, y_offset), font, font_scale, color, thickness)
        #     y_offset += 25
        
        # # Vẽ banner cảnh báo nếu có vật cản
        # if dangerous_objects:
        #     alert_height = 100
        #     cv2.rectangle(frame_copy, (0, 0), (w, alert_height), (0, 0, 255), -1)
        #     cv2.rectangle(frame_copy, (0, 0), (w, alert_height), (0, 255, 255), 4)
            
        #     alert_text = f"🚨 DANGER! {len(dangerous_objects)} OBSTACLE(S) DETECTED FOR >3 SECONDS"
        #     font_scale_large = 1.0
        #     thickness_large = 2
        #     text_size = cv2.getTextSize(alert_text, font, font_scale_large, thickness_large)[0]
        #     x_text = (w - text_size[0]) // 2
            
        #     cv2.putText(frame_copy, alert_text, (x_text, 60),
        #                font, font_scale_large, (255, 255, 255), thickness_large)
        
        return frame_copy

