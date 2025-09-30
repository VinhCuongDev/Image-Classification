# Phân Loại Ảnh Sử Dụng Mạng Nơ-Ron Tích Chập

## Giới Thiệu

Phân loại ảnh là một bài toán quan trọng trong lĩnh vực xử lý ảnh số, sử dụng mạng nơ-ron tích chập (CNN) để phân loại các hình ảnh vào các lớp khác nhau. Dự án này được viết bằng Python và sử dụng thư viện TensorFlow nhằm mục đích:

-  Hiểu cách thức hoạt động của CNN trong - phân loại ảnh.
- Áp dụng CNN để phân loại các tập dữ liệu ảnh thực tế.
- Đánh giá hiệu suất của mô hình CNN được xây dựng.

## Tập Dữ Liệu
Được lấy từ Kaggle: https://www.kaggle.com/datasets/puneet6060/intel-image-classification
Tập dữ liệu được lấy từ các cảnh thiên nhiên trên thế giới. Dữ liệu chứa khoảng 25k ảnh có kích thước 150x150 gồm các mục là buildings, forest, glacier, mountain, sea, street
## Mô Hình CNN

        model = Sequential()
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=self.input_shape))
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(200, kernel_size=(3, 3), activation='relu'))
        model.add(Conv2D(300, kernel_size=(3, 3), activation='relu'))
        model.add(Conv2D(200, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(4, 4)))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(self.classnum, activation='softmax'))

## Kết Quả

-  Accuracy: 0.98
-  Epochs : 50
-  Batch_size: 32
-  Kết Quả Ghi Nhận Được:


![model_accuracy](https://i.imgur.com/xydPWH3.png)
![model_loss](https://i.imgur.com/phPsydk.png)
## Cài Đặt 
- Python = 3.10.11
- Bước 1:
  
    - git clone https://github.com/HVINHCuong/Classification.git
    - cd Classification
- Bước 2:
    run python app.py

## Video Demo
![video_demo](https://i.imgur.com/Sfhkhg2.gif)



