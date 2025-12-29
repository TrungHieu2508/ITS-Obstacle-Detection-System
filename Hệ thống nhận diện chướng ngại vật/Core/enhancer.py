import cv2
import numpy as np

class ImageEnhancer:
    def __init__(self, use_gamma=True, gamma=1.3, use_hist=False):
        self.use_gamma = use_gamma
        self.gamma = gamma
        self.use_hist = use_hist

    def gamma_correction(self, image):
        inv = 1.0 / self.gamma
        table = np.array([
            ((i / 255.0) ** inv) * 255
            for i in np.arange(256)
        ]).astype("uint8")
        return cv2.LUT(image, table)

    def histogram_equalization(self, image):
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        y = cv2.equalizeHist(y)
        ycrcb = cv2.merge((y, cr, cb))
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    def process(self, frame):
        if self.use_gamma:
            frame = self.gamma_correction(frame)
        if self.use_hist:
            frame = self.histogram_equalization(frame)
        return frame
