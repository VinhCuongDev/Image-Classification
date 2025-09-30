from image_proces import ImagePreprocessor
from datapreprocessor import DataPreprocessor
from cnnmodel import MyCNNModel

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import joblib

# Khởi tạo một thể hiện của lớp DataPreprocessor
data_preprocessor = DataPreprocessor()
# Gọi phương thức load_data để tải dữ liệu từ thư mục 'dataset'
X_test, Y_test = data_preprocessor.load_data('dataset\\seg_test')
X_train, Y_train = data_preprocessor.load_data('dataset\\seg_train')


encoder = LabelEncoder()
Y_train = encoder.fit_transform(Y_train)
Y_test = encoder.transform(Y_test)
joblib.dump(encoder,'class.joblib')
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)

X_test,X_train,Y_test,Y_train = np.array(X_test),np.array(X_train),np.array(Y_test),np.array(Y_train)\

X_test, X_train = np.expand_dims(X_test,axis=-1), np.expand_dims(X_train,axis=-1)

model = MyCNNModel(input_shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3]), classnum=len(encoder.classes_))
model.summary()
model.fit(X_train,Y_train,batch_size=32,epochs=30,validation_data=(X_test,Y_test))
model.save('model.h5')
model.plot_training_history()