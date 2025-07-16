import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# Define dataset path
dataset_path = "dataset"  # Path to your dataset

# Data augmentation for training dataset
data_augmentation = keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
])

# Load dataset with 80% training and 20% validation split
train_dataset = keras.utils.image_dataset_from_directory(
    dataset_path,
    image_size=(128, 128),  # Resize images to 128x128
    batch_size=32,
    validation_split=0.2,
    subset="training",
    seed=42
)

val_dataset = keras.utils.image_dataset_from_directory(
    dataset_path,
    image_size=(128, 128),
    batch_size=32,
    validation_split=0.2,
    subset="validation",
    seed=42
)

# Get class names (gesture labels)
class_names = train_dataset.class_names
print(f"Classes: {class_names}")

# Normalize pixel values (0 to 1 range)
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.map(lambda x, y: (x / 255.0, y))  # Normalize images
val_dataset = val_dataset.map(lambda x, y: (x / 255.0, y))  # Normalize images

# Prefetch for better performance
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)

# Build CNN Model
model = keras.Sequential([
    layers.InputLayer(input_shape=(128, 128, 3)),  # Input shape
    data_augmentation,  # Data augmentation
    layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(256, (3, 3), activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Dropout(0.5),  # Adding dropout for regularization
    layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Dropout(0.5),  # Adding dropout for regularization
    layers.Dense(len(class_names), activation='softmax')  # Output layer
])

# Compile the model
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.00001),  # Lower learning rate
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Add early stopping to prevent overfitting
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,  # Increase patience
    restore_best_weights=True
)

# Compute class weights
labels = np.concatenate([y for x, y in train_dataset], axis=0)
class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weights_dict = dict(enumerate(class_weights))

# Train the model
history = model.fit(train_dataset,
                    validation_data=val_dataset,
                    epochs=50,
                    class_weight=class_weights_dict,
                    callbacks=[early_stopping])

# Save the model
model.save("hand_gesture_model_3.h5")
print("Model saved as hand_gesture_model_3.h5")