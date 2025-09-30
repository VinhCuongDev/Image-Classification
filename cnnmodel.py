from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt


class MyCNNModel:
    def __init__(self,input_shape = None, classnum = None):
        self.input_shape = input_shape
        self.classnum = classnum
        self.model = self.build_model()
        self.history = None
    def build_model(self):
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
        return model
    
    def summary(self):
        self.model.summary()

    def fit(self, X, Y, batch_size=32, epochs=10, validation_data=None):
        if self.classnum> 2:
            self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        else:
            self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.history = self.model.fit(X, Y, batch_size=batch_size, epochs=epochs, validation_data=validation_data)
        
    def save(self):
        self.model.save('model.h5')

    def plot_training_history(self):
        if not self.history:
            print("Model history not available. Please train the model first using the 'fit' function.")
            return
        # Vẽ biểu đồ cho độ chính xác và loss của tập huấn luyện và tập validation
        plt.plot(self.history.history['accuracy'])
        plt.plot(self.history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()
        
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()
