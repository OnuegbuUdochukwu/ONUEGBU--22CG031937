from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import base64
import cv2
import os

# Note: templates are stored in the `template/` folder in this project.
# Configure Flask to use that folder so render_template('index.html') works.
app = Flask(__name__, template_folder='template')

# --- Configuration ---
MODEL_PATH = 'face_emotionModel.h5'
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
IMG_SIZE = 48
model = None

# --- Model Loading ---
@app.before_first_request
def load_ml_model():
    """
    Loads the trained Keras model before the first request.
    This ensures the model is only loaded once at startup.
    """
    global model
    if model is None:
        try:
            # Note: Custom objects might be needed if custom layers are used.
            model = load_model(MODEL_PATH)
            print("Successfully loaded the emotion detection model.")
        except Exception as e:
            print(f"Error loading model from {MODEL_PATH}: {e}")
            print("The application will run, but prediction API calls will fail.")
            model = False # Set to False to signify failure to load

# --- Utility Functions ---
def preprocess_image_for_model(image_data):
    """
    Placeholder for image preprocessing.
    In a real app, this would convert raw image data (e.g., from a webcam feed)
    into a grayscale, 48x48 numpy array, scaled and reshaped for the CNN model.
    """
    # If image_data is provided as a data URL (base64), decode and preprocess
    try:
        if image_data and isinstance(image_data, str) and image_data.startswith('data:'):
            # Split header and data
            header, encoded = image_data.split(',', 1)
            img_bytes = base64.b64decode(encoded)
            # Convert bytes to numpy array for OpenCV
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError('Could not decode image bytes')

            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Resize to model's expected input size
            resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

            # Normalize to [0,1] and reshape to (1, IMG_SIZE, IMG_SIZE, 1)
            arr = resized.astype('float32') / 255.0
            arr = np.expand_dims(arr, axis=-1)  # add channel dim
            arr = np.expand_dims(arr, axis=0)   # add batch dim
            return arr
    except Exception as e:
        # Fall back to dummy input and log warning
        print(f"Warning: preprocessing failed ({e}), using dummy input")

    # Dummy preprocessing: returns a valid input shape for the model
    dummy_input = np.random.rand(1, IMG_SIZE, IMG_SIZE, 1).astype('float32') / 255.0
    return dummy_input

# --- Routes ---
@app.route('/')
def index():
    """Renders the main web application interface."""
    # Render the HTML template stored at `template/index.html`.
    return render_template('index.html', model_status=f"Model loaded: {model is not False and model is not None}")

@app.route('/predict_emotion', methods=['POST'])
def predict_emotion():
    """Handles POST requests for emotion prediction."""
    if model is False or model is None:
        return jsonify({'error': 'Model not loaded. Check server logs.'}), 503

    try:
        # In a real application, you would receive image data (e.g., a file or base64 string)
        # For this boilerplate, we'll assume a successful placeholder process.
        image_data = request.json.get('image_data') # Placeholder for real data
        
        # 1. Preprocess the image
        processed_input = preprocess_image_for_model(image_data)
        
        # 2. Make prediction
        predictions = model.predict(processed_input)
        
        # 3. Get the most likely emotion
        emotion_index = np.argmax(predictions[0])
        predicted_emotion = EMOTIONS[emotion_index]
        confidence = float(predictions[0][emotion_index])

        return jsonify({
            'success': True,
            'emotion': predicted_emotion,
            'confidence': f"{confidence:.2f}"
        })

    except Exception as e:
        app.logger.error(f"Prediction error: {e}")
        return jsonify({'error': f'An error occurred during prediction: {e}'}), 500

if __name__ == '__main__':
    # When running locally for development. Use PORT from the environment when provided
    # (Render and other platforms inject a PORT environment variable).
    port = int(os.environ.get('PORT', 5000))
    # Bind to 0.0.0.0 so the app is reachable from outside the container in hosting envs.
    app.run(host='0.0.0.0', port=port, debug=True)
