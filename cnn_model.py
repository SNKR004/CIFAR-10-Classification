import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# Load data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize
x_train = x_train / 255.0
x_test = x_test / 255.0  
print(x_train.shape, x_test.shape)

# CNN architecture
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 classes for CIFAR-10
])

# Compiling
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Training & Evaluation
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Prediction
predictions = model.predict(x_test)

predicted_class = tf.argmax(predictions[0]).numpy()
actual_class = y_test[0][0]

print(f'Actual class for the first test image: {actual_class}')
print(f'Predicted class for the first test image: {predicted_class}')

# Save
model.save('cifar10_cnn_model.keras')
