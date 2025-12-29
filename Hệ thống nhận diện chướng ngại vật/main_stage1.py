import cv2
import config
from core.camera import Camera
from core.enhancer import ImageEnhancer
from core.fps import FPSCounter
from core.dashboard import Dashboard
from core.recorder import VideoRecorder

def main():
    cam = Camera(
        config.CAMERA_SOURCE,
        config.FRAME_WIDTH,
        config.FRAME_HEIGHT
    )

    if not cam.is_opened():
        print("❌ Không mở được camera")
        return

    enhancer = ImageEnhancer(
        config.USE_GAMMA,
        config.GAMMA_VALUE,
        config.USE_HIST_EQ
    )

    fps_counter = FPSCounter(config.FPS_BUFFER_SIZE)
    dashboard = Dashboard()

    recorder = VideoRecorder(
        config.VIDEO_OUTPUT,
        config.VIDEO_FPS,
        (config.FRAME_WIDTH, config.FRAME_HEIGHT)
    )

    if config.RECORD_VIDEO:
        recorder.start()

    while True:
        frame = cam.read()
        if frame is None:
            break

        frame = enhancer.process(frame)
        fps_avg = fps_counter.update()
        frame = dashboard.draw(frame, fps_avg, recorder.recording)

        recorder.write(frame)

        cv2.imshow("Obstacle Detection System - Stage 1", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite("snapshot_stage1.jpg", frame)
            print("📸 Snapshot saved")

    recorder.stop()
    recorder.release()
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
