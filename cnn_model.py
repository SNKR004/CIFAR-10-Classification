#Libraries Used

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Data Preperation

#Split the data into training and testing sets, i.e. 50,000 and 10,000 respectively
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalize the images, Scale pixel values from the range [0 to 255] to [0.0 to 1.0]
x_train, x_test = x_train / 255.0, x_test / 255.0 

# Data Visualization

fig, ax = plt.subplots(5, 5)
k = 0
for i in range(5):
	for j in range(5):
		ax[i][j].imshow(x_train[k], aspect='auto')
		k += 1
plt.show()

# CNN Architecture

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

# Model Compilation

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Training
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Evaluating
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'\nTest accuracy: {test_acc}')

# Prediction
predictions = model.predict(x_test)

# Validation
predicted_class = tf.argmax(predictions[0]).numpy()
print(f'Predicted class for the first test image: {predicted_class}')

# Save the model
model.save('cifar10_cnn_model.keras')
