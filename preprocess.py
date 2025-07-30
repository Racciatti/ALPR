import cv2
import numpy as np
from skimage.morphology import skeletonize

class PreprocessingPipeline:
    def __init__(self, image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.image = image.astype(np.uint8)

    def resize(self, width=100, height=75):
        self.image = cv2.resize(self.image, (width, height))
        return self

    def binarize(self, threshold=128):
        _, self.image = cv2.threshold(self.image, threshold, 255, cv2.THRESH_BINARY)
        return self

    def adaptive_threshold(self):
        self.image = cv2.adaptiveThreshold(self.image, 255, 
                                           cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 11, 2)
        return self

    def gaussian_blur(self, ksize=3):
        self.image = cv2.GaussianBlur(self.image, (ksize, ksize), 0)
        return self

    def median_blur(self, ksize=3):
        self.image = cv2.medianBlur(self.image, ksize)
        return self

    def histogram_equalization(self):
        self.image = cv2.equalizeHist(self.image)
        return self

    def skeletonize(self):
        binary = (self.image > 127).astype(np.uint8)
        self.image = (skeletonize(binary) * 255).astype(np.uint8)
        return self

    def dilate(self, ksize=3, iterations=1):
        kernel = np.ones((ksize, ksize), np.uint8)
        self.image = cv2.dilate(self.image, kernel, iterations=iterations)
        return self

    def erode(self, ksize=3, iterations=1):
        kernel = np.ones((ksize, ksize), np.uint8)
        self.image = cv2.erode(self.image, kernel, iterations=iterations)
        return self

    def normalize(self):
        self.image = self.image.astype(np.float32) / 255.0
        return self

    def get(self):
        return self.image