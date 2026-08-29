from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np

# Example dataset placeholders (replace with real data)
X_train = np.random.rand(100, 128, 128, 1)
y_train = np.random.randint(0, 2, 100)
X_val   = np.random.rand(20, 128, 128, 1)
y_val   = np.random.randint(0, 2, 20)

# Define CNN
cnn_model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train CNN
cnn_model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Save model
cnn_model.save("saved_models/cnn_model.keras")

print("✅ CNN model trained and saved to saved_models/cnn_model.keras")
