from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np
from keras import layers
import tensorflow as tf
import os
import keras
import csv
import matplotlib.pyplot as plt

folder_test = './validation'
mapa_labeluri_test = {}

# Definim modelul
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
# Incarcam modelul
model.load_weights('save_at_1_model4_64_32_16_8.keras')

# Procesam imaginile de test
imagini_test = []
for img_path in os.listdir(folder_test):
    if not img_path.endswith('.png'):
        continue
    print(folder_test + '/' + img_path)
    img = image.load_img(folder_test + '/' + img_path
        , target_size=(200, 200))
    imagini_test.append([img, img_path])
# Incarcam imaginea si o redimensionam
def incarca_imagine(file_path):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, (200, 200))
    image = tf.expand_dims(image, axis=0)
    return image
    
# Prezicem etichetele pentru imaginile de test
for img, img_path in imagini_test:
    if not img_path.endswith('.png'):
        continue
    img = incarca_imagine(folder_test + '/' + img_path)
    prediction = model.predict(img)
    mapa_labeluri_test[img_path] = np.argmax(prediction)
    print(f'{img_path}: {np.argmax(prediction)}')
    # plt.imshow(img)
    plt.show()
        
# Scriem rezultatele in fisier
with open('results_model4_64_32_16_8_f.csv', mode='w') as results_file:
    scrie_rez = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    scrie_rez.writerow(['image_id', 'label'])
    for key in mapa_labeluri_test:
        scrie_rez.writerow([
            key.split('.')[0], mapa_labeluri_test[key]])