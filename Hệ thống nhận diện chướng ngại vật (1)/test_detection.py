import cv2
from yolo_detector import YOLOObstacleDetector

# Test file
video_path = "Phathienchuongngaivat.mp4"

print("[TEST] Khởi tạo detector...")
detector = YOLOObstacleDetector()

print(f"[TEST] Mở video: {video_path}")
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("[ERROR] Không mở được video!")
    exit()

frame_count = 0
total_detections = 0

print("[TEST] Đọc 10 frame đầu...\n")

while frame_count < 10:
    ret, frame = cap.read()
    if not ret:
        print("[INFO] Hết video")
        break
    
    frame_count += 1
    print(f"\n{'='*60}")
    print(f"Frame {frame_count}: {frame.shape}")
    
    # Test với confidence rất thấp
    for test_conf in [0.5, 0.3, 0.1]:
        frame_test, detections = detector.detect(frame, conf=test_conf)
        
        print(f"  Confidence {test_conf}: {len(detections)} detections", end="")
        if detections:
            for det in detections:
                print(f"\n    - {det['class_name']}: {det['confidence']:.2%}", end="")
            total_detections += len(detections)
        print()

cap.release()
print(f"\n{'='*60}")
print(f"[SUMMARY] Total: {frame_count} frames, {total_detections} objects found")
print(f"[RESULT] Nếu tìm thấy vật → tăng confidence")
print(f"[RESULT] Nếu không tìm → model không nhận diện video này")
