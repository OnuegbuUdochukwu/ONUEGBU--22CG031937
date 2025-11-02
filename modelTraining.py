import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import os

# Define file paths
DATASET_PATH = 'data/emotions_dataset.csv'
MODEL_SAVE_PATH = 'face_emotionModel.h5'
IMG_SIZE = 48
BATCH_SIZE = 64
EPOCHS = 2  # quick verification run (reduced from 50)

def load_and_preprocess_data():
    """Loads the dataset and prepares it for training."""
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset not found at {DATASET_PATH}. Please ensure the CSV file is placed there.")
        return None, None
    # Load the data (expecting columns: 'emotion','pixels','Usage')
    data = pd.read_csv(DATASET_PATH)

    required_cols = {'emotion', 'pixels', 'Usage'}
    if not required_cols.issubset(set(data.columns)):
        raise ValueError(f"CSV must contain the columns: {required_cols}. Found: {set(data.columns)}")

    print(f"Data loaded successfully. Total samples: {len(data)}")

    # Helper to convert pixel string to numpy array
    def pixels_to_array(pixels_str):
        # Use fromstring for speed; pixels are space-separated integers
        arr = np.fromstring(pixels_str, dtype=np.uint8, sep=' ')
        if arr.size != IMG_SIZE * IMG_SIZE:
            # If something odd, try splitting
            arr = np.array([int(p) for p in pixels_str.split()])
        arr = arr.reshape((IMG_SIZE, IMG_SIZE, 1)).astype('float32') / 255.0
        return arr

    # Preallocate lists for split sets
    splits = {
        'Training': {'X': [], 'y': []},
        'PublicTest': {'X': [], 'y': []},
        'PrivateTest': {'X': [], 'y': []}
    }

    # Iterate rows and build arrays
    for idx, row in data.iterrows():
        usage = row['Usage']
        if usage not in splits:
            # skip unknown usages
            continue
        pixels = row['pixels'] if 'pixels' in row else row['Pixels']
        emotion = int(row['emotion'])
        try:
            img = pixels_to_array(pixels)
        except Exception as e:
            # skip malformed rows but report
            print(f"Skipping row {idx} due to pixel parse error: {e}")
            continue
        splits[usage]['X'].append(img)
        splits[usage]['y'].append(emotion)

    # Convert lists to numpy arrays and one-hot encode labels
    X_train = np.stack(splits['Training']['X'], axis=0) if len(splits['Training']['X']) > 0 else np.empty((0, IMG_SIZE, IMG_SIZE, 1))
    y_train = np.array(splits['Training']['y'], dtype=np.int32)

    X_val = np.stack(splits['PublicTest']['X'], axis=0) if len(splits['PublicTest']['X']) > 0 else np.empty((0, IMG_SIZE, IMG_SIZE, 1))
    y_val = np.array(splits['PublicTest']['y'], dtype=np.int32)

    X_test = np.stack(splits['PrivateTest']['X'], axis=0) if len(splits['PrivateTest']['X']) > 0 else np.empty((0, IMG_SIZE, IMG_SIZE, 1))
    y_test = np.array(splits['PrivateTest']['y'], dtype=np.int32)

    # One-hot encode labels
    num_classes = 7
    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes)
    y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes)
    y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes)

    print(f"Train samples: {len(X_train)}, Val samples: {len(X_val)}, Test samples: {len(X_test)}")

    return (X_train, y_train_cat), (X_val, y_val_cat), (X_test, y_test_cat)

def build_cnn_model(num_classes=7):
    """Defines the Convolutional Neural Network (CNN) architecture."""
    model = Sequential([
        # First Block
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        # Second Block
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        # Third Block
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        # Classification Head
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model():
    """Trains the model and saves the weights."""
    print("Starting model training setup...")
    datasets = load_and_preprocess_data()

    if datasets is None:
        return

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = datasets

    # Data generators: training with augmentation, validation/test only rescale
    train_datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Note: inputs already in range [0,1] from preprocessing. ImageDataGenerator will not rescale again.
    train_generator = train_datagen.flow(x=X_train, y=y_train, batch_size=BATCH_SIZE)

    val_datagen = ImageDataGenerator()
    val_generator = val_datagen.flow(x=X_val, y=y_val, batch_size=BATCH_SIZE, shuffle=False)

    model = build_cnn_model(num_classes=7)
    model.summary()

    # Callbacks for better training
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    ]

    steps_per_epoch = max(1, len(X_train) // BATCH_SIZE)
    validation_steps = max(1, len(X_val) // BATCH_SIZE)

    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        validation_data=val_generator,
        validation_steps=validation_steps,
        callbacks=callbacks
    )

    # After training, evaluate on test set
    if len(X_test) > 0:
        # Create a test datagen (no augmentation)
        test_datagen = ImageDataGenerator()
        test_generator = test_datagen.flow(x=X_test, y=y_test, batch_size=BATCH_SIZE, shuffle=False)
        test_steps = max(1, len(X_test) // BATCH_SIZE)
        test_loss, test_acc = model.evaluate(test_generator, steps=test_steps, verbose=1)
        print(f"Final Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")
    else:
        print("No test set available to evaluate.")

    # Ensure model saved (ModelCheckpoint already saved best model)
    if not os.path.exists(MODEL_SAVE_PATH):
        model.save(MODEL_SAVE_PATH)
        print(f"Model saved to {MODEL_SAVE_PATH}")
    else:
        print(f"Best model already saved to {MODEL_SAVE_PATH}")


if __name__ == '__main__':
    # This block executes when model_training.py is run directly
    train_model()
