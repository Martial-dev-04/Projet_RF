import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers
from utils import build_embedding_model  # même dossier src/

# Paramétrage
DATA_DIR = 'C:/PROJETS/RF_dataset/DatasetOriginal_clean_aug'              # dossier contenant les images classées par personne
SPLIT_DIR = 'C:/PROJETS/RF_dataset/DatasetOriginal_splits'       # généré par data_preparation.py
MODEL_DIR = '../models'
IMG_SIZE = 160
BATCH_SIZE = 32
EPOCHS_HEAD = 10
EPOCHS_FINE = 20
LEARNING_RATE = 5e-4

# Charger les listes train/val
def load_image_paths_and_labels(txt_file):
    paths, labels = [], []
    label_to_id = {}
    with open(txt_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            img_path, label = parts[0], parts[1]
            if label not in label_to_id:
                label_to_id[label] = len(label_to_id)
            paths.append(img_path)
            labels.append(label_to_id[label])
    return np.array(paths), np.array(labels), label_to_id

train_paths, train_labels, label2id = load_image_paths_and_labels(os.path.join(SPLIT_DIR, 'train.txt'))
val_paths, val_labels, _ = load_image_paths_and_labels(os.path.join(SPLIT_DIR, 'val.txt'))
num_classes = len(label2id)
id2label = {v: k for k, v in label2id.items()}

# Data pipeline
def parse_image(img_path, label):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32) / 255.0
    return img, label

def create_dataset(paths, labels, batch_size, training=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    if training:
        # Ajouter augmentation de données
        def augment(img, label):
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_flip_up_down(img)
            img = tf.image.random_brightness(img, 0.2)
            img = tf.image.random_contrast(img, 0.8, 1.2)
            # Rotation légère
            img = tf.image.rot90(img, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
            return img, label
        
        ds = ds.shuffle(10000)  # Buffer plus grand
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = create_dataset(train_paths, train_labels, BATCH_SIZE, training=True)
val_ds = create_dataset(val_paths, val_labels, BATCH_SIZE)

# Modèle complet (embedding + softmax)
model = build_embedding_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=num_classes)

# Phase 1 : entraîner seulement la tête de classification
model.compile(
    optimizer=optimizers.Adam(LEARNING_RATE),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
print("Phase 1 : Entraînement de la tête de classification...")
# ✅ Callbacks pour éviter le surapprentissage
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,  # Arrête si val_loss ne s'améliore pas en 5 epochs
        restore_best_weights=True,
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        os.path.join(MODEL_DIR, 'best_finetuned_model.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=0
    )
]

model.fit(
    train_ds, 
    validation_data=val_ds, 
    epochs=EPOCHS_HEAD,
    callbacks=callbacks  # ✅ Ajouter
)

# Phase 2 : fine-tuning des 30 dernières couches du backbone
base_model = model.layers[1]  # MobileNetV2
base_model.trainable = True
# Gèle toutes les couches sauf les 30 dernières
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Recompile avec un learning rate plus faible
model.compile(
    optimizer=optimizers.Adam(1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
print("Phase 2 : Fine‑tuning...")
model.fit(
    train_ds, 
    validation_data=val_ds, 
    epochs=EPOCHS_FINE,
    callbacks=callbacks  # ✅ Réutiliser
)

# Sauvegarde du modèle complet (on peut plus tard extraire l'embedding)
model.save(os.path.join(MODEL_DIR, 'myFace_model.h5'))

# Extraire et sauvegarder uniquement le modèle d'embedding
embedding_model = keras.Model(model.input, model.get_layer('l2_norm').output)
embedding_model.save(os.path.join(MODEL_DIR, 'myFace_embedding_model.h5'))
print("Modèle d'embedding sauvegardé dans", MODEL_DIR)