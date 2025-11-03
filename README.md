# 😊 Facial Emotion Detection System

[![Python](https://img.shields.io/badge/Python-3.10.14-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20+-orange.svg)](https://www.tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-2.2.5-green.svg)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A real-time facial emotion detection web application powered by deep learning. This system uses a Convolutional Neural Network (CNN) to classify facial expressions into seven emotion categories with high accuracy.

![Emotion Detection Demo](https://img.shields.io/badge/Status-Active-success)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Technologies Used](#-technologies-used)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Training](#-model-training)
- [Deployment](#-deployment)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## 🎯 Overview

This Facial Emotion Detection System is a machine learning-powered web application that detects and classifies human emotions from facial expressions in real-time. The system can identify seven distinct emotional states:

- 😠 **Angry**
- 🤢 **Disgust**
- 😨 **Fear**
- 😊 **Happy**
- 😢 **Sad**
- 😲 **Surprise**
- 😐 **Neutral**

The application uses a webcam feed to capture facial images, processes them through a trained deep learning model, and displays the predicted emotion with confidence scores in real-time.

### Purpose

This project was developed as a comprehensive machine learning application demonstrating:
- Computer vision techniques for facial analysis
- Deep learning model architecture design and training
- Full-stack web development with real-time processing
- Production deployment of ML models

---

## ✨ Features

### Core Functionality
- 🎥 **Real-time Webcam Integration**: Capture live video feed from user's camera
- 🧠 **Deep Learning Model**: Custom-trained CNN for emotion classification
- ⚡ **Fast Predictions**: Low-latency inference for real-time user experience
- 📊 **Confidence Scores**: Display prediction confidence for transparency
- 📱 **Responsive Design**: Modern, mobile-friendly interface built with Tailwind CSS

### Technical Features
- 🔄 **Image Preprocessing Pipeline**: Automatic grayscale conversion, resizing, and normalization
- 🎨 **Data Augmentation**: Enhanced model robustness through augmented training
- 💾 **Model Checkpointing**: Save best models during training
- 📈 **Performance Monitoring**: Callbacks for learning rate reduction and early stopping
- 🚀 **Production Ready**: Configured for deployment on cloud platforms (Render, Heroku, etc.)

---

## 🛠 Technologies Used

### Machine Learning & AI
- **TensorFlow/Keras** (≥2.20.0) - Deep learning framework
- **OpenCV** (4.7.0) - Computer vision and image processing
- **NumPy** (1.26.4) - Numerical computing
- **Pandas** (2.1.0) - Data manipulation and analysis
- **scikit-learn** (1.4.2) - Machine learning utilities

### Web Framework
- **Flask** (2.2.5) - Python web framework
- **Gunicorn** (20.1.0) - WSGI HTTP server for production

### Frontend
- **HTML5** - Structure
- **Tailwind CSS** - Styling and responsive design
- **JavaScript** - Webcam integration and API communication

### Data Visualization (Training)
- **Matplotlib** (3.7.1) - Plotting and visualization
- **Seaborn** (0.12.2) - Statistical data visualization

### Additional Tools
- **Pillow** (9.4.0) - Image processing
- **tqdm** (4.65.0) - Progress bars for training

---

## 📁 Project Structure

```
ONUEGBU--22CG031937/
├── app.py                      # Main Flask application
├── modelTraining.py            # Model training script
├── face_emotionModel.h5        # Pre-trained emotion detection model
├── requirements.txt            # Python dependencies
├── runtime.txt                 # Python version specification
├── .gitignore                  # Git ignore rules
│
├── template/
│   └── index.html             # Web interface template
│
├── EmotionDetector-001/
│   └── Procfile               # Deployment configuration
│
└── data/                      # Dataset directory (not tracked in git)
    └── emotions_dataset.csv   # Training data (FER2013 or similar)
```

### Key Files Description

- **`app.py`**: Flask web application with routes for the main interface and prediction API
- **`modelTraining.py`**: Complete pipeline for loading data, building CNN, and training the model
- **`face_emotionModel.h5`**: Serialized trained Keras model (48x48 grayscale input)
- **`template/index.html`**: Modern web interface with webcam integration
- **`requirements.txt`**: All Python package dependencies with pinned versions

---

## 🚀 Installation

### Prerequisites

- **Python**: Version 3.10.14 (specified in `runtime.txt`)
- **pip**: Python package manager
- **Webcam**: For real-time emotion detection (optional for testing)
- **Git**: For cloning the repository

### Step-by-Step Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/OnuegbuUdochukwu/ONUEGBU--22CG031937.git
   cd ONUEGBU--22CG031937
   ```

2. **Create a Virtual Environment** (Recommended)
   ```bash
   # On Windows
   python -m venv venv
   venv\Scripts\activate

   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

   > **Note for macOS Apple Silicon Users**: For better performance, consider using `tensorflow-macos` and `tensorflow-metal` instead of the standard TensorFlow package. Uncomment the relevant lines in `requirements.txt`.

4. **Verify Installation**
   ```bash
   python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
   python -c "import cv2; print('OpenCV version:', cv2.__version__)"
   ```

---

## 💻 Usage

### Running the Application Locally

1. **Start the Flask Server**
   ```bash
   python app.py
   ```

   The application will start on `http://0.0.0.0:5000` by default.

2. **Access the Web Interface**
   - Open your web browser
   - Navigate to `http://localhost:5000`
   - Allow camera permissions when prompted

3. **Using the Application**
   - The webcam feed will appear on the screen
   - Click the **"Capture & Predict Emotion"** button
   - The detected emotion and confidence score will be displayed
   - Try different facial expressions to see different predictions!

### API Endpoint

The application exposes a REST API for emotion prediction:

**Endpoint**: `POST /predict_emotion`

**Request Body**:
```json
{
  "image_data": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAA..."
}
```

**Response**:
```json
{
  "success": true,
  "emotion": "Happy",
  "confidence": "0.95"
}
```

**Example cURL Request**:
```bash
curl -X POST http://localhost:5000/predict_emotion \
  -H "Content-Type: application/json" \
  -d '{"image_data": "data:image/jpeg;base64,...base64_encoded_image..."}'
```

---

## 🎓 Model Training

The project includes a complete pipeline for training your own emotion detection model.

### Dataset Requirements

The training script expects a CSV file with the following structure:

- **Location**: `data/emotions_dataset.csv`
- **Columns**:
  - `emotion`: Integer label (0-6 representing the 7 emotions)
  - `pixels`: Space-separated pixel values (2304 values for 48x48 images)
  - `Usage`: Dataset split indicator (`Training`, `PublicTest`, or `PrivateTest`)

**Recommended Dataset**: [FER2013 (Facial Expression Recognition 2013)](https://www.kaggle.com/datasets/msambare/fer2013)

### Training Process

1. **Prepare the Dataset**
   ```bash
   # Create data directory if it doesn't exist
   mkdir -p data
   
   # Place your emotions_dataset.csv in the data/ folder
   # The data/ folder is gitignored to avoid committing large files
   ```

2. **Run Training**
   ```bash
   python modelTraining.py
   ```

3. **Training Configuration**
   - **Image Size**: 48x48 pixels (grayscale)
   - **Batch Size**: 64
   - **Epochs**: 2 (default for quick verification; increase to 50+ for production)
   - **Optimizer**: Adam
   - **Loss Function**: Categorical Cross-Entropy

4. **Model Architecture**
   ```
   Conv2D(32) → MaxPool → Dropout(0.25)
   Conv2D(64) → MaxPool → Dropout(0.25)
   Conv2D(128) → MaxPool → Dropout(0.25)
   Flatten
   Dense(256) → Dropout(0.5)
   Dense(7, softmax)
   ```

5. **Training Features**
   - **Data Augmentation**: Rotation, shifting, shearing, zooming, and horizontal flipping
   - **Callbacks**:
     - Model checkpointing (saves best model based on validation accuracy)
     - Learning rate reduction on plateau
     - Early stopping to prevent overfitting

6. **Output**
   - Trained model saved as `face_emotionModel.h5`
   - Training metrics displayed in console
   - Final test set evaluation

### Customizing Training

Edit the constants in `modelTraining.py`:

```python
IMG_SIZE = 48        # Image dimensions
BATCH_SIZE = 64      # Batch size
EPOCHS = 50          # Number of training epochs
```

---

## 🌐 Deployment

The application is configured for easy deployment to cloud platforms.

### Deployment on Render

1. **Connect Repository**: Link your GitHub repository to Render
2. **Configure Build**:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 2`
3. **Environment Variables**: Set `PORT` if not auto-configured
4. **Deploy**: Render will automatically deploy your application

### Deployment on Heroku

1. **Install Heroku CLI**: [Download here](https://devcenter.heroku.com/articles/heroku-cli)

2. **Login and Create App**:
   ```bash
   heroku login
   heroku create your-emotion-detector
   ```

3. **Deploy**:
   ```bash
   git push heroku main
   ```

4. **Open Application**:
   ```bash
   heroku open
   ```

### Using the Procfile

The `EmotionDetector-001/Procfile` contains the deployment configuration:
```
web: gunicorn app:app --bind 0.0.0.0:$PORT --workers 2
```

This tells the platform how to run your application in production.

### Environment Configuration

- **Runtime**: Python version specified in `runtime.txt`
- **Dependencies**: Installed from `requirements.txt`
- **Port**: Automatically assigned by the hosting platform via `$PORT` environment variable

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help improve this project:

### How to Contribute

1. **Fork the Repository**
   ```bash
   # Click the "Fork" button on GitHub
   git clone https://github.com/your-username/ONUEGBU--22CG031937.git
   ```

2. **Create a Feature Branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```

3. **Make Your Changes**
   - Write clean, documented code
   - Follow existing code style and conventions
   - Add comments for complex logic

4. **Test Your Changes**
   ```bash
   # Run the application and verify functionality
   python app.py
   ```

5. **Commit Your Changes**
   ```bash
   git add .
   git commit -m "Add amazing feature"
   ```

6. **Push to Your Fork**
   ```bash
   git push origin feature/amazing-feature
   ```

7. **Open a Pull Request**
   - Go to the original repository on GitHub
   - Click "New Pull Request"
   - Describe your changes in detail

### Contribution Guidelines

- **Code Style**: Follow PEP 8 for Python code
- **Documentation**: Update README if adding new features
- **Testing**: Ensure changes don't break existing functionality
- **Commits**: Use clear, descriptive commit messages

### Areas for Improvement

- 🎯 Add unit tests and integration tests
- 📊 Implement visualization of model performance metrics
- 🌍 Add support for multiple languages
- 🎨 Enhance UI/UX with animations and better feedback
- 🔒 Add authentication and user sessions
- 📸 Support image upload in addition to webcam
- 🎭 Expand to detect more emotions or micro-expressions
- ⚡ Optimize model for mobile deployment (TensorFlow Lite)

---

## 📄 License

This project is licensed under the **MIT License**.

### MIT License Summary

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

**THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED.**

For the full license text, see the [LICENSE](LICENSE) file in the repository.

---

## 🙏 Acknowledgments

### Datasets
- **FER2013**: Facial Expression Recognition dataset used for training
- Kaggle community for dataset accessibility and support

### Technologies & Frameworks
- **TensorFlow Team**: For the powerful deep learning framework
- **Flask Contributors**: For the lightweight and flexible web framework
- **OpenCV Community**: For comprehensive computer vision tools
- **Tailwind CSS**: For modern, utility-first CSS framework

### Inspiration & Resources
- Research papers on facial emotion recognition and CNNs
- Online tutorials and courses on deep learning and computer vision
- Open-source ML projects that demonstrate best practices

### Special Thanks
- **Onuegbu Udochukwu** - Project Creator and Developer
- Academic advisors and mentors who provided guidance
- Beta testers who provided valuable feedback

---

## 📧 Contact & Support

### Developer
**Onuegbu Udochukwu**
- GitHub: [@OnuegbuUdochukwu](https://github.com/OnuegbuUdochukwu)

### Support
If you encounter any issues or have questions:
1. Check existing [Issues](https://github.com/OnuegbuUdochukwu/ONUEGBU--22CG031937/issues)
2. Create a new issue with detailed description
3. Include error messages, screenshots, and system information

### Reporting Bugs
When reporting bugs, please include:
- Operating system and version
- Python version
- Error messages and stack traces
- Steps to reproduce the issue
- Expected vs. actual behavior

---

## 🌟 Show Your Support

If you find this project helpful or interesting:
- ⭐ Star the repository
- 🍴 Fork it for your own use
- 📢 Share it with others
- 🐛 Report bugs
- 💡 Suggest new features

---

## 📊 Project Statistics

![GitHub repo size](https://img.shields.io/github/repo-size/OnuegbuUdochukwu/ONUEGBU--22CG031937)
![GitHub last commit](https://img.shields.io/github/last-commit/OnuegbuUdochukwu/ONUEGBU--22CG031937)
![GitHub issues](https://img.shields.io/github/issues/OnuegbuUdochukwu/ONUEGBU--22CG031937)
![GitHub stars](https://img.shields.io/github/stars/OnuegbuUdochukwu/ONUEGBU--22CG031937)

---

<div align="center">

**Made with ❤️ and 🧠 by Onuegbu Udochukwu**

*Empowering machines to understand human emotions*

</div>
