import cv2
import os
import numpy as np
from image_proces import ImagePreprocessor# Import class ImagePreprocessor từ file image_process.py


class DataPreprocessor:
    def __init__(self, image_size=(224, 224)):
        self.preprocessor = ImagePreprocessor(image_size=image_size)
        
    def load_data(self, folder):
        data_images = []
        labels = []
        dir_path = os.listdir(folder)
        # Duyệt qua mỗi thư mục con
        for label in dir_path:
            label_directory = os.path.join(folder, label)

            # Duyệt qua mỗi tệp ảnh trong thư mục con
            for filename in os.listdir(label_directory):
                filepath = os.path.join(label_directory, filename)
                # Kiểm tra nếu tệp đó là ảnh
                if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpg'):
                    try:
                        image = cv2.imread(filepath)
                        image = self.preprocessor.resize_image(image,image_size=(50,50))
                        image = self.preprocessor.convert_to_gray(image)
                        image = self.preprocessor.apply_gaussian_blur(image, kernel_size=(3,3))
                        image = self.preprocessor.canny_edge_detection(image)
                        image_normalize = self.preprocessor.normalize_image(image)
                        data_images.append(image_normalize)
                        labels.append(label)
                        '''
                        for i in range(2):
                            image_a = self.preprocessor.augment_image(image)
                            image_n = self.preprocessor.normalize_image(image_a)
                            data_images.append(image_n)
                            labels.append(label)
                        '''
                    except Exception as e:
                        print("Lỗi khi đọc {}: {}".format(filename, str(e)))

        return data_images, labels
    

    def save_data(self, X, Y, X_filename='X_data.npy', Y_filename='Y_data.npy'):
        with open(X_filename, 'wb') as f:
            np.save(f, X)
        with open(Y_filename, 'wb') as f:
            np.save(f, Y) 