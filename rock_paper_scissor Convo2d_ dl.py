

!pip install -q kaggle
from google.colab import files
files.upload()
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d ekaterinalaiona/rock-paper-scissors
!unzip rock-paper-scissors.zip -d rock_paper_scissors

import pandas as pd
import tensorflow_datasets as tfds

df = tfds.load('rock_paper_scissors',as_supervised=True)
#Bunu da kullanabiliriz ama biraz kaggle pratiği yapmak istedim.
# csupeagle
#df

!ls -l rock_paper_scissors

import os

train_dir = "rock_paper_scissors/Rock_Paper_Scissors/train_rps"
test_dir = "rock_paper_scissors/Rock_Paper_Scissors/test_rps"

print("Train directory contents:", os.listdir(train_dir))
print("Test directory contents:", os.listdir(test_dir))

for category in os.listdir(train_dir):
    print(f"Contents of {category}: {os.listdir(os.path.join(train_dir, category))}")

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split
import numpy as np
import os

train_dir = "rock_paper_scissors/Rock_Paper_Scissors/train_rps/train_rps"
test_dir = "rock_paper_scissors/Rock_Paper_Scissors/test_rps/test_rps"

def load_images_from_directory(directory, target_size=(150, 150)):
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory '{directory}' not found")
    images = []
    labels = []
    class_names = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    for idx, class_name in enumerate(class_names):
        class_dir = os.path.join(directory, class_name)
        for image_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_name)
            if os.path.isfile(image_path):
                img = image.load_img(image_path, target_size=target_size)
                img = image.img_to_array(img)
                images.append(img)
                labels.append(idx)
    return np.array(images), np.array(labels), class_names

train_images, train_labels, class_names = load_images_from_directory(train_dir)
print(f"Loaded {len(train_images)} training images.")

test_images, test_labels, _ = load_images_from_directory(test_dir)
print(f"Loaded {len(test_images)} testing images.")

if len(train_images) > 0:
    train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.15, random_state=42)
else:
    raise ValueError("No training images found.")

train_images = train_images / 255.
val_images = val_images / 255.
test_images = test_images / 255.

print("Sınıf adları:", class_names)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
# Basit bir CNN modeli oluşturma
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPool2D(2, 2),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPool2D(2, 2),
    Flatten(),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(len(class_names), activation='softmax')
])

learning_rate = 0.0015
optimizer = Adam(learning_rate=learning_rate)

model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    train_images, train_labels,
    validation_data=(val_images, val_labels),
    epochs=5,
    batch_size=32
)

test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f"Test doğruluğu: {test_accuracy * 100:.2f}%")

plt.figure(figsize=(10, 8))

plt.subplot(2, 1, 1)
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend(loc='lower right')
plt.title('Eğitim ve Doğrulama Doğruluğu')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.xlabel('Epoch')
plt.ylabel('Kayıp')
plt.legend(loc='upper right')
plt.title('Eğitim ve Doğrulama Kaybı')

plt.tight_layout()
plt.show()

history_dict = history.history
epochs = range(1, len(history_dict['accuracy']) + 1)


data = {'epoch': epochs,
    'accuracy': history_dict['accuracy'],
    'val_accuracy': history_dict['val_accuracy'],
    'loss': history_dict['loss'],
    'val_loss': history_dict['val_loss']}


df = pd.DataFrame(data)

# Eğitim ve doğrulama doğruluğunu gösteren grafik

plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='epoch', y='accuracy', label='Train Accuracy')
sns.lineplot(data=df, x='epoch', y='val_accuracy', label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Train and Validation Accuracy')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='epoch', y='loss', label='Train Loss')
sns.lineplot(data=df, x='epoch', y='val_loss', label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train and Validation Losses')
plt.legend()
plt.grid(True)
plt.show()