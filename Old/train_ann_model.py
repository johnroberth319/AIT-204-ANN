import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import numpy as np

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values to range [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Flatten the images for ANN (28x28 = 784 features)
x_train_flat = x_train.reshape(-1, 28 * 28)
x_test_flat = x_test.reshape(-1, 28 * 28)

print(f"Training data shape: {x_train_flat.shape}")
print(f"Test data shape: {x_test_flat.shape}")

# Create the ANN model
model = models.Sequential([
    # Input layer - 784 neurons (28x28 flattened)
    layers.Dense(512, activation='relu', input_shape=(784,)),
    layers.Dropout(0.2),  # Prevent overfitting
    
    # Hidden layer 1
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.2),
    
    # Hidden layer 2
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.1),
    
    # Output layer - 10 neurons for digits 0-9
    layers.Dense(10, activation='softmax')
])

# Display model architecture
model.summary()

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
print("Training the ANN model...")
history = model.fit(
    x_train_flat, y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.1,
    verbose=1
)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test_flat, y_test, verbose=0)
print(f'Test accuracy: {test_acc:.4f}')

# Save the model
model.save('mnist_ann_model.h5')
print("ANN model saved as 'mnist_ann_model.h5'")

# Print comparison info
print(f"\n{'='*50}")
print("ANN Model Summary:")
print(f"Architecture: Fully Connected Neural Network")
print(f"Input: 784 features (28x28 flattened)")
print(f"Hidden Layers: 512 -> 256 -> 128 neurons")
print(f"Output: 10 classes (digits 0-9)")
print(f"Total Parameters: {model.count_params():,}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"{'='*50}")