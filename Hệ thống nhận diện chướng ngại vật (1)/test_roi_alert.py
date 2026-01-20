import cv2
from yolo_detector import YOLOObstacleDetector

# File test ROI + Cảnh báo
video_path = "Phathienchuongngaivat.mp4"

print("[SETUP] Khởi tạo detector với ROI...")
detector = YOLOObstacleDetector()

# ===== ĐỊNH NGHĨA ROI =====
# VÍ DỤ 1: ROI hình chữ nhật (vùng ở giữa frame)
# Giả sử frame ~1280x720, ROI ở giữa: x1=400, y1=250, x2=880, y2=550
# detector.set_roi_rectangle(400, 250, 880, 550)

# VÍ DỤ 2: ROI dạng đa giác (vùng hình tam giác)
roi_points = [
    (640, 100),   # Đỉnh trên
    (200, 500),   # Góc trái dưới
    (1080, 500)   # Góc phải dưới
]
detector.set_roi_polygon(roi_points)

print(f"\n[TEST] Mở video: {video_path}")
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("[ERROR] Không mở được video!")
    exit()

frame_count = 0
roi_alerts = 0

print("[TEST] Đọc 30 frame và kiểm tra ROI...\n")

while frame_count < 30:
    ret, frame = cap.read()
    if not ret:
        print("[INFO] Hết video")
        break
    
    frame_count += 1
    
    # Phát hiện + kiểm tra ROI + vẽ cảnh báo
    frame_result, detections, detections_in_roi = detector.detect_with_roi(
        frame, 
        conf=0.4,
        show_roi=True,      # Vẽ ROI lên frame
        show_alert=True     # Hiển thị cảnh báo nếu có vật trong ROI
    )
    
    # In thông tin chi tiết
    print(f"{'='*70}")
    print(f"Frame {frame_count}:")
    print(f"  - Tất cả vật phát hiện: {len(detections)}")
    print(f"  - Vật trong ROI: {len(detections_in_roi)}")
    
    if detections_in_roi:
        roi_alerts += 1
        print(f"  ⚠️  CẢNH BÁO! Phát hiện vật trong vùng nguy hiểm:")
        for det in detections_in_roi:
            print(f"      • {det['class_name']:15} | Confidence: {det['confidence']:5.1%} | " +
                  f"ROI Overlap: {det['roi_overlap']:.0f}%")
    
    # Hiển thị frame (tùy chọn - comment nếu không muốn hiển thị)
    # cv2.imshow('Detection with ROI', frame_result)
    # key = cv2.waitKey(1)
    # if key == ord('q'):
    #     break

cap.release()
cv2.destroyAllWindows()

print(f"\n{'='*70}")
print(f"[SUMMARY] Tổng: {frame_count} frames, {roi_alerts} frame có cảnh báo")
print(f"[INFO] ROI type: {detector.roi_type}")
if detector.roi is not None:
    print(f"[INFO] ROI coordinates: {detector.roi}")
