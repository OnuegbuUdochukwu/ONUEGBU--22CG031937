from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import os

app = Flask(__name__)

# --- Configuration ---
MODEL_PATH = 'face_emotionModel.h5'
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
IMG_SIZE = 48
model = None

# --- Model Loading ---
@app.before_request
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
    # Dummy preprocessing: returns a valid input shape for the model
    # Replace this with actual OpenCV/PIL image handling
    dummy_input = np.random.rand(1, IMG_SIZE, IMG_SIZE, 1).astype('float32') / 255.0
    return dummy_input

# --- Routes ---
@app.route('/')
def index():
    """Renders the main web application interface."""
    return render_template('index.htm', model_status=f"Model loaded: {model is not False and model is not None}")

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
    # When running locally for development
    app.run(debug=True, port=5000)
