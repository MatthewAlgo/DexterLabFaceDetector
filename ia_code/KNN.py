import os
import numpy as np
import csv
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from tensorflow import data as tf_data

image_size = (200, 200)
batch_size = 128

# Citim labelurile pentru datele de antrenare si validare
map_labeluri_train = {}
map_labeluri_validare = {}
with open('train.csv', mode='r') as infile:
    reader = csv.reader(infile)
    for rows in reader:
        if rows[1] == 'label':
            continue
        map_labeluri_train[rows[0]] = int(rows[1])
with open('validation.csv', mode='r') as infile:
    reader = csv.reader(infile)
    for rows in reader:
        if rows[1] == 'label':
            continue
        map_labeluri_validare[rows[0]] = int(rows[1])

fold_train = './train'
fold_validare = './validation'
paths_pentru_train = []
labels_pentru_train = []
paths_pentru_validare = []
labeluri_pentru_validare = []
for filename, label in map_labeluri_train.items():
    file_path = fold_train + '/' + filename + '.png'
    if os.path.exists(file_path):
        paths_pentru_train.append(file_path)
        labels_pentru_train.append(label)
for filename, label in map_labeluri_validare.items():
    file_path = fold_validare + '/' + filename + '.png'
    if os.path.exists(file_path):
        paths_pentru_validare.append(file_path)
        labeluri_pentru_validare.append(label)
def load_image(file_path, label):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, image_size)
    # Tranformam in grayscale si aplicam filtrul Sobel
    image = tf.image.rgb_to_grayscale(image)
    image = tf.expand_dims(image, axis=0)
    sobel_edges = tf.image.sobel_edges(image)
    sobel_x = sobel_edges[..., 0]
    sobel_y = sobel_edges[..., 1]
    sobel = tf.sqrt(tf.square(sobel_x) + tf.square(sobel_y))
    # Scoatem dimensiunea de batch
    sobel = tf.squeeze(sobel, axis=0)
    return sobel, label

# Formam dataseturile
dataset_antrenare = tf_data.Dataset.from_tensor_slices((paths_pentru_train, labels_pentru_train))
dataset_antrenare = dataset_antrenare.map(load_image)
dataset_antrenare = dataset_antrenare.batch(batch_size)
dataset_antrenare = dataset_antrenare.shuffle(buffer_size=1024)
# Prefetch pentru a incarca datele mai rapid
dataset_antrenare = dataset_antrenare.prefetch(buffer_size=tf_data.experimental.AUTOTUNE)

dataset_validare = tf_data.Dataset.from_tensor_slices((paths_pentru_validare, labeluri_pentru_validare))
dataset_validare = dataset_validare.map(load_image)
dataset_validare = dataset_validare.batch(batch_size)
dataset_validare = dataset_validare.shuffle(buffer_size=1024)
dataset_validare = dataset_validare.prefetch(buffer_size=tf_data.experimental.AUTOTUNE)

# Preprocesare, rotire imagine random si luminozitatea si contrastul random
def preprocesare(images, labels):
    images = tf.image.random_flip_left_right(images)
    images = tf.image.random_flip_up_down(images)
    images = tf.image.random_brightness(images, max_delta=0.1)
    images = tf.image.random_contrast(images, lower=0.9, upper=1.1)
    return images, labels

dataset_antrenare = dataset_antrenare.map(preprocesare)
dataset_antrenare = dataset_antrenare.prefetch(buffer_size=tf_data.AUTOTUNE)
dataset_antrenare = dataset_antrenare.map(lambda x, y: (x / 255.0, y))

plt.figure(figsize=(10, 10))
for images, labels in dataset_antrenare.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().squeeze(), cmap='gray')
        plt.title(np.argmax(labels[i].numpy()))
        plt.axis("off")
plt.show()

# Knn cu 3 vecini
knn = KNeighborsClassifier(n_neighbors=10, n_jobs=-1, weights='distance', metric='euclidean')

imagini_train = []
labeluri_train = []
for images, labels in dataset_antrenare:
    for i in range(images.shape[0]):
        imagini_train.append(images[i].numpy().flatten())
        labeluri_train.append(np.argmax(labels[i].numpy()))
imagini_train = np.array(imagini_train)
labeluri_train = np.array(labeluri_train)
# Facem fit pe datele de antrenare
knn.fit(imagini_train, labeluri_train)

# Predictii pe validare
folder_valid = './validation'
mapa_labeluri_valid = {}
imagini_valid = []
for img_path in os.listdir(folder_valid):
    img = tf.io.read_file(folder_valid + '/' + img_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, image_size)
    # Convertim imaginea in grayscale si aplicam sobel
    img = tf.image.rgb_to_grayscale(img)
    img = tf.expand_dims(img, axis=0)
    margini_sobel = tf.image.sobel_edges(img)
    sobel_x = margini_sobel[..., 0]
    sobel_y = margini_sobel[..., 1]
    sobel = tf.sqrt(tf.square(sobel_x) + tf.square(sobel_y))
    sobel = tf.squeeze(sobel, axis=0).numpy().flatten()
    imagini_valid.append(sobel)
imagini_valid = np.array(imagini_valid)
valid_labeluri = knn.predict(imagini_valid)

for img_path, label in zip(os.listdir(folder_valid), valid_labeluri):
    mapa_labeluri_valid[img_path] = label
with open('valid_labeluri.csv', mode='w') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(['filename', 'label'])
    for filename, label in mapa_labeluri_valid.items():
        writer.writerow([filename, label])