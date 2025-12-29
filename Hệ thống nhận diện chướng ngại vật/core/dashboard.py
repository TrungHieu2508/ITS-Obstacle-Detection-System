import cv2
import datetime

class Dashboard:
    def draw(self, frame, fps, recording):
        h, w = frame.shape[:2]
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        lines = [
            f"FPS(avg): {int(fps)}",
            f"Resolution: {w}x{h}",
            f"Time: {timestamp}",
            f"Mode: STAGE 1 - CAMERA",
            f"Recording: {'ON' if recording else 'OFF'}"
        ]

        x = 20
        y = 40
        line_gap = 30

        for line in lines:
            cv2.putText(
                frame,
                line,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,  # Font mảnh, dễ nhìn
                0.8,                       # Kích thước vừa
                (0, 0, 255),               # Màu đỏ
                2,                         # Không in đậm
                cv2.LINE_AA
            )
            y += line_gap

        return frame
