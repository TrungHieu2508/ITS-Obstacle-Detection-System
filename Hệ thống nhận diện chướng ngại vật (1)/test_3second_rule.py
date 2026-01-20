import cv2
from yolo_detector import YOLOObstacleDetector

"""
QUY TẮC 3 GIÂY:
- 0s đến 2.9s: Vật thể xuất hiện, đếm thời gian
- Từ 3.1s trở đi: Nếu vật thể vẫn còn trong vùng quét (ROI) -> VẬT CẢN NGUY HIỂM
"""

video_path = "Phathienchuongngaivat.mp4"

print("[SETUP] Khởi tạo detector với quy tắc 3 giây...")
detector = YOLOObstacleDetector()

# Định nghĩa ROI
roi_points = [
    (640, 100),
    (200, 500),
    (1080, 500)
]
detector.set_roi_polygon(roi_points)
detector.reset_tracking()

print(f"[TEST] Mở video: {video_path}")
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("[ERROR] Không mở được video!")
    exit()

# Lấy FPS từ video
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"[INFO] Video FPS: {fps}")

frame_count = 0
total_dangerous_frames = 0

print(f"\n[TEST] Đọc video và áp dụng quy tắc 3 giây...\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[INFO] Hết video")
        break
    
    # Phát hiện chướng ngại vật
    frame_detected, detections = detector.detect(frame, conf=0.4)
    
    # Kiểm tra vật cản trong ROI
    detections_in_roi = detector.get_detections_in_roi(detections)
    
    # ===== UPDATE TRACKING & KIỂM TRA QUY TẮC 3 GIÂY =====
    tracking_result = detector.update_tracking(detections_in_roi, frame_count, fps=fps)
    
    dangerous_objects = tracking_result['dangerous_objects']
    time_elapsed = tracking_result['time_elapsed']
    
    # In thông tin chi tiết
    print(f"\n{'='*80}")
    print(f"Frame {frame_count} (Time: {frame_count/fps:.2f}s):")
    print(f"  Detections: {len(detections)} | In ROI: {len(detections_in_roi)} | " +
          f"Tracked: {tracking_result['total_tracked']}")
    
    # Hiển thị tracking info
    if time_elapsed:
        print(f"\n  📊 TRACKING INFO:")
        for obj_id in sorted(time_elapsed.keys()):
            elapsed = time_elapsed[obj_id]
            obj_info = detector.tracked_objects[obj_id]
            
            if elapsed > detector.time_threshold_seconds:
                print(f"    ⚠️  ID {obj_id}: {obj_info['class_name']:12} | " +
                      f"Time: {elapsed:.2f}s | Status: 🚨 DANGEROUS!")
                total_dangerous_frames += 1
            else:
                print(f"    ✅ ID {obj_id}: {obj_info['class_name']:12} | " +
                      f"Time: {elapsed:.2f}s | Status: Tracking...")
    
    # Vẽ thông tin tracking lên frame
    frame_detected = detector.draw_roi(frame_detected)
    frame_detected = detector.draw_tracking_info(frame_detected, tracking_result)
    
    # Hiển thị frame (tùy chọn)
    # cv2.imshow('3-Second Rule Detection', frame_detected)
    # key = cv2.waitKey(1)
    # if key == ord('q'):
    #     break
    
    frame_count += 1
    
    # Dừng sau 100 frames để test
    if frame_count >= 100:
        break

cap.release()
cv2.destroyAllWindows()

print(f"\n{'='*80}")
print(f"[SUMMARY]")
print(f"  Total frames: {frame_count}")
print(f"  FPS: {fps:.2f}")
print(f"  Total time: {frame_count/fps:.2f}s")
print(f"  Frames with dangerous objects: {total_dangerous_frames}")
print(f"\n[RULE] Nếu Time > 3.0s và vật thể vẫn còn trong ROI -> NGUY HIỂM!")
