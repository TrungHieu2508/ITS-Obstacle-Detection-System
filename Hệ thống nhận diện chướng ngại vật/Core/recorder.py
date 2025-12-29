import cv2

class VideoRecorder:
    def __init__(self, filename, fps, size):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(filename, fourcc, fps, size)
        self.recording = False

    def start(self):
        self.recording = True

    def stop(self):
        self.recording = False

    def write(self, frame):
        if self.recording:
            self.writer.write(frame)

    def release(self):
        self.writer.release()
