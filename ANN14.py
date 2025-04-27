
#4. MNIST Handwritten Character Detection using PyTorch, Keras and Tensorflow


# Import required libraries
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# ================== Keras (TensorFlow) Model ===========================
# Load and preprocess MNIST dataset using TensorFlow (Keras)
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# Normalize the pixel values to 0-1 range
train_images = train_images / 255.0
test_images = test_images / 255.0

# Expand dimensions to fit CNN input [batch, height, width, channels]
train_images = train_images[..., tf.newaxis]
test_images = test_images[..., tf.newaxis]

# Build a simple CNN model using Keras
keras_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')  # 10 classes for digits 0-9
])

# Compile the Keras model
keras_model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

# Train the Keras model
keras_history = keras_model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))

# Evaluate the Keras model on test data
keras_test_loss, keras_test_acc = keras_model.evaluate(test_images, test_labels, verbose=2)
print(f'\nKeras Test Accuracy: {keras_test_acc}')

# ================== PyTorch Model ===========================
# Load and preprocess MNIST dataset using PyTorch
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# Define the CNN model for PyTorch
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64*5*5, 128)  # Adjust size after pooling
        self.fc2 = nn.Linear(128, 10)  # 10 classes (0-9 digits)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64*5*5)  # Flattening layer
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize model, loss function, and optimizer for PyTorch
pytorch_model = CNNModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(pytorch_model.parameters(), lr=0.001)

# Train the PyTorch model
epochs = 5
for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in trainloader:
        optimizer.zero_grad()
        outputs = pytorch_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(trainloader):.4f}')

# Evaluate the PyTorch model on test data
correct = 0
total = 0
with torch.no_grad():
    for images, labels in testloader:
        outputs = pytorch_model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'PyTorch Test Accuracy: {100 * correct / total:.2f}%')

# ================== Visualization of Predictions =========================
# Visualizing predictions for Keras
keras_predictions = keras_model.predict(test_images)

# Plot first 5 test images with Keras predictions
for i in range(5):
    plt.imshow(test_images[i].reshape(28, 28), cmap=plt.cm.binary)
    plt.title(f"Keras Predicted: {tf.argmax(keras_predictions[i]).numpy()}, Actual: {test_labels[i]}")
    plt.show()

# Visualizing predictions for PyTorch
dataiter = iter(testloader)
images, labels = dataiter.next()

# Get PyTorch predictions
outputs = pytorch_model(images)
_, predicted = torch.max(outputs, 1)

# Plot first 5 test images with PyTorch predictions
for i in range(5):
    plt.imshow(images[i].numpy().squeeze(), cmap='gray')
    plt.title(f"PyTorch Predicted: {predicted[i].item()}, Actual: {labels[i]}")
    plt.show()
