import tensorflow as tf
import os
import numpy as np
import keras
from keras import layers
from keras.optimizers import RMSprop, Adam
from tensorflow.python.client import device_lib
from tensorflow import data as tf_data
import matplotlib.pyplot as plt
import csv
import pandas as pd
from tensorflow.python.client import device_lib

# Verificam daca avem GPU
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
print(device_lib.list_local_devices())
def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]

print(get_available_devices())
# Dimensiunea imaginilor si batch size-ul
image_size = (200, 200)
batch_size = 128

mapa_labeluri_train = {}
mapa_labeluri_validare = {}
with open('train.csv', mode='r') as infile:
    reader = csv.reader(infile)
    for rows in reader:
        if rows[1] == 'label':
            continue
        mapa_labeluri_train[rows[0]] = int(rows[1])
with open('validation.csv', mode='r') as infile:
    reader = csv.reader(infile)
    for rows in reader:
        if rows[1] == 'label':
            continue
        mapa_labeluri_validare[rows[0]] = int(rows[1])
for filename, label in mapa_labeluri_validare.items():
    print(f'{filename}: {label}')

fold_train = './train'
fold_validare = './validation'
paths_pentru_train = []
labels_pentru_train = []
paths_pentru_validare = []
labeluri_pentru_validare = []

for filename, label in mapa_labeluri_train.items():
    file_path = fold_train + '/' + filename + '.png'
    if os.path.exists(file_path): 
        paths_pentru_train.append(file_path)
        labels_pentru_train.append(label)
for filename, label in mapa_labeluri_validare.items():
    file_path = fold_validare + '/' + filename + '.png'
    if os.path.exists(file_path): 
        paths_pentru_validare.append(file_path)
        labeluri_pentru_validare.append(label)

def load_image(file_path, label):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, image_size)    
    return image, label
# Crearea dataseturilor
setdate_train = tf_data.Dataset.from_tensor_slices((paths_pentru_train, labels_pentru_train))
# Incarcam imaginile si le redimensionam
setdate_train = setdate_train.map(load_image)
# Amestecam datele si le incarcam in batch-uri
setdate_train = setdate_train.batch(batch_size)
setdate_train = setdate_train.shuffle(buffer_size=1024)
# Incarcam datele in memorie
setdate_train = setdate_train.prefetch(buffer_size=tf_data.experimental.AUTOTUNE)
# La fel si pentru validare
setdate_validare = tf_data.Dataset.from_tensor_slices((paths_pentru_validare, labeluri_pentru_validare))
setdate_validare = setdate_validare.map(load_image)
setdate_validare = setdate_validare.batch(batch_size)
setdate_validare = setdate_validare.shuffle(buffer_size=1024)
setdate_validare = setdate_validare.prefetch(buffer_size=tf_data.experimental.AUTOTUNE)

# Preprocesare, unde rotim imaginile, le schimbam luminozitatea si contrastul
def preprocesare(imagini, labels_pentru_train):
    imagini = tf.image.random_flip_left_right(imagini)
    imagini = tf.image.random_flip_up_down(imagini)
    imagini = tf.image.random_brightness(imagini, max_delta=0.1)
    imagini = tf.image.random_contrast(imagini, lower=0.9, upper=1.1)
    return imagini, labels_pentru_train

augumented_setdate_train = setdate_train.map(preprocesare)
# Normalizare
augumented_setdate_train = augumented_setdate_train.map(lambda x, y: (x/255, y))
setdate_train = setdate_train.map(lambda x, y: (x/255, y))
setdate_train = setdate_train.map(lambda image, label: preprocesare(image, label))
setdate_train = setdate_train.prefetch(buffer_size=tf_data.AUTOTUNE)

# plt.figure(figsize=(10, 10))
# for imagini, labels_pentru_train in setdate_train.take(1):
#     for i in range(9):
#         ax = plt.subplot(3, 3, i + 1)
#         plt.imshow(imagini[i])
#         plt.title(labels_pentru_train[i].numpy())
#         plt.axis("off")
# plt.show()

def creare_model(input_shape):
    input_sh = keras.Input(shape=input_shape)
    # Modelul are 3 layere de convolutie, fiecare urmata de un max pooling
    x = layers.Conv2D(128, (3, 3), activation='relu', input_shape=input_shape)(input_sh) # 128 filtre, kernel de 3x3
    x = layers.MaxPooling2D((2, 2))(x) # Pooling de 2x2
    x = layers.Conv2D(64, (3, 3), activation='relu')(x) # 64 filtre, kernel de 3x3
    x = layers.Conv2D(64, (3, 3), activation='relu')(x) # 64 filtre, kernel de 3x3
    x = layers.MaxPooling2D((2, 2))(x) # Pooling de 2x2
    x = layers.Conv2D(32, (3, 3), activation='relu')(x) # 32 filtre, kernel de 3x3
    x = layers.Conv2D(32, (3, 3), activation='relu')(x) # 32 filtre, kernel de 3x3
    x = layers.MaxPooling2D((2, 2))(x) # Pooling de 2x2
    x = layers.Conv2D(16, (3, 3), activation='relu')(x) # 16 filtre, kernel de 3x3
    x = layers.Conv2D(8, (3, 3), activation='relu')(x) # 8 filtre, kernel de 3x3
    x = layers.MaxPooling2D((2, 2))(x) # Pooling de 2x2
    x = layers.BatchNormalization(momentum=0.9)(x) # Normalizare batch
    x = layers.Flatten()(x) # Adunam toate feature-urile intr-un singur vector
    outputs = layers.Dense(3, activation='softmax')(x) # 3 neuroni pentru cele 3 clase
    return keras.Model(input_sh, outputs)

model = creare_model(input_shape=(200, 200, 3))
# Cream modelul pentru 100 de epoci
epoci = 100
callbacks = [
    # La fiecare epoca, salvam modelul
    keras.callbacks.ModelCheckpoint("save_at_{epoch}_model4_64_32_16_8.keras"),
]
# Compilare
model.compile(
    optimizer=keras.optimizers.Adam(0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
# Antrenare
model.fit(
    setdate_train, epochs=epoci, callbacks=callbacks, validation_data=setdate_validare,
)