from flask import Flask, render_template, request, jsonify
from werkzeug.datastructures import FileStorage
import tensorflow as tf
import cv2
import numpy as np
import os
import joblib


from image_proces import ImagePreprocessor



app = Flask(__name__)
app.config['STATIC_FOLDER'] = 'static'  # Define static folder path
app.config['UPLOAD_FOLDER'] = os.path.join(app.config['STATIC_FOLDER'], 'images')

process_image = ImagePreprocessor()
model = tf.keras.models.load_model('model.h5')
label_encoder = joblib.load('class.joblib')

# Check if the upload folder exists, if not, create it
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/')
def index():
    return render_template('index.html')  # Render the main page

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Read image from the request directly
        image_file = request.files['image']
        if image_file  and allowed_file(image_file.filename):

            image_array = read_image(image_file)
            # Preprocess the image
            image = preprocessing_image(image_array)
            image = np.array(image)
            image = np.expand_dims(image, axis=-1)
            print(image.shape)
            image = image.reshape(1,50,50,-1)
            # Predict
            prediction = model.predict(image)
            predictions = []
            for i, prob in enumerate(prediction.flatten()):
                class_names = label_encoder.inverse_transform([i])[0]
                percentages = prob * 100
                predictions.append({'class_name': class_names, 'percentage': percentages})

            return jsonify(predictions)

def read_image(file_storage: FileStorage) -> np.ndarray:
    """Read image from FileStorage and return as numpy array."""
    file_bytes = np.frombuffer(file_storage.read(), np.uint8)
    image_array = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return image_array

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

def preprocessing_image(image):
    image = process_image.resize_image(image, image_size=(50, 50))
    image = process_image.convert_to_gray(image)
    image = process_image.apply_gaussian_blur(image, kernel_size=(3, 3))
    image = process_image.canny_edge_detection(image)
    image = process_image.normalize_image(image)
    return image

if __name__ == '__main__':
    app.run(debug=True)