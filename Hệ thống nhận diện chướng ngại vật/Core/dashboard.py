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

        y = 30
        for line in lines:
            cv2.putText(frame, line, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 255), 2)
            y += 25

        return frame
