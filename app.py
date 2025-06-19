from flask import Flask, redirect, render_template, request
from PIL import Image
import numpy as np
import tensorflow as tf
import os

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('C:\\Users\\FreeComp\\Wiza_Brain.keras')


# Define class mappings
class_mappings = {0: ('Glioma', 'Glioma is a growth of cells that starts in the brain or spinal cord. The cells in a glioma look similar to healthy brain cells called glial cells.'),
                  1: ('Meningioma', 'Meningioma is a tumor that grows from the membranes that surround the brain and spinal cord, called the meninges.'),
                  2: ('Notumor', 'No tumor detected.'),
                  3: ('Pituitary', 'Pituitary tumors are tumors that form in the pituitary gland near the brain. These tumors can cause changes in hormone levels.')}

# Function to preprocess and load an image
def load_and_preprocess_image(image_path, image_shape=(168, 168)):
    img = Image.open(image_path).convert("L")  # Convert to grayscale
    img = img.resize(image_shape)  # Resize to match model input size
    img_array = np.array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Function to predict tumor type and confidence score
def predict_tumor(image_path):
    img_array = load_and_preprocess_image(image_path)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    predicted_class_name, description = class_mappings[predicted_class]
    confidence_score = np.max(prediction)  # Confidence score
    return predicted_class_name, description, confidence_score

@app.route('/')
def start():
    return render_template('start.html')

@app.route('/predict', methods=['GET', 'POST'])  # Accept both GET and POST requests
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', message='No file part')

        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', message='No selected file')

        try:
            # Get the directory path of the Flask application
            app_dir = os.path.dirname(os.path.abspath(__file__))

            # Specify the path where you want to save the image temporarily
            file_path = os.path.join(app_dir, 'static', 'temp_image.jpg')

            # Save the file temporarily
            file.save(file_path)

            predicted_class, description, confidence_score = predict_tumor(file_path)
            return render_template('index.html', message=f'Predicted Tumor Type: {predicted_class}', description=description, image_path='static/temp_image.jpg', predicted_class=predicted_class, confidence_score=confidence_score)
        except Exception as e:
            return render_template('index.html', message='Error occurred during prediction')
    else:
        # Redirect to the start page if accessing /predict via GET request
        return redirect('/')

if __name__ == '__main__':
    app.run(debug=True)

print("http://127.0.0.1:5000")
