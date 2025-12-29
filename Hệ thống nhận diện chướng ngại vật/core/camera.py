import cv2

class Camera:
    def __init__(self, source, width, height):
        self.cap = cv2.VideoCapture(source)
        self.width = width
        self.height = height

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def is_opened(self):
        return self.cap.isOpened()

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return cv2.resize(frame, (self.width, self.height))

    def release(self):
        self.cap.release()
