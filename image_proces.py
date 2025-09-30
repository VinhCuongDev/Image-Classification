import cv2
import numpy as np
from imgaug import augmenters as iaa

class ImagePreprocessor:
    def __init__(self, image_size=(224, 224)):
        self.image_size = image_size

    def resize_image(self, image, image_size=None):
        if image_size is None:
            image_size = self.image_size
        return cv2.resize(image, image_size)

    def convert_to_gray(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def apply_gaussian_blur(self, image, kernel_size=(5, 5)):
        return cv2.GaussianBlur(image, kernel_size, 0)

    def canny_edge_detection(self, image, low_threshold=50, high_threshold=150):
        return cv2.Canny(image, low_threshold, high_threshold)
    
    def normalize_image(self, image):
        return image/255.0

    def augment_image(self, image):
        seq = iaa.Sequential([
            iaa.Fliplr(0.5),  # Lật ảnh theo chiều ngang với xác suất 0.5
            iaa.Affine(rotate=(-20, 20)),  # Xoay ảnh trong khoảng -20 đến 20 độ
            iaa.GaussianBlur(sigma=(0, 3.0)),  # Áp dụng Gaussian Blur với độ lớn từ 0 đến 3.0
            iaa.AdditiveGaussianNoise(scale=(0, 0.05*255))  # Thêm nhiễu Gaussian với độ lớn từ 0 đến 0.05*255
        ], random_order=True)  # Thực hiện các phép biến đổi ngẫu nhiên theo thứ tự ngẫu nhiên
        augmented_image = seq(image=image)
        return augmented_image
