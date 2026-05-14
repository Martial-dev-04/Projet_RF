import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2

# -------------------------------------------------------------------
# Construction du modèle SANS la couche Lambda (la normalisation L2
# sera faite manuellement après l'inférence)
# -------------------------------------------------------------------
def build_embedding_model(input_shape=(160, 160, 3), num_classes=None):
    """
    Version d'entraînement (peut inclure une tête softmax).
    Pour l'inférence, on utilisera la fonction ci-dessous.
    """
    base_model = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    base_model.trainable = False

    inputs = keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.BatchNormalization(name='bn_pre_embedding')(x)
    x = layers.Dropout(0.3, name='dropout_pre_embedding')(x)
    x = layers.Dense(
        128,
        activation=None,
        kernel_regularizer=keras.regularizers.l2(1e-4),
        name='embedding'
    )(x)
    x = layers.BatchNormalization(name='bn_post_embedding')(x)

    # Pas de Lambda ici : on normalise à la main après la prédiction
    embeddings = x  # Ce vecteur sera normalisé en dehors du modèle

    if num_classes is not None:
        x = layers.Dropout(0.4, name='dropout_classifier')(embeddings)
        classifier = layers.Dense(
            num_classes,
            activation='softmax',
            kernel_regularizer=keras.regularizers.l2(1e-4),
            name='classifier'
        )(x)
        model = keras.Model(inputs, classifier)
    else:
        model = keras.Model(inputs, embeddings)

    return model


def build_inference_model_from_weights(h5_path, input_shape=(160, 160, 3)):
    """
    Reconstruit le modèle d'inférence (sans Lambda) à partir des poids sauvés.
    On utilise la même architecture que l'entraînement mais sans la tête softmax,
    et on ignore la couche Lambda manquante.
    """
    # 1. Construire le modèle d'embedding (sans softmax)
    inference_model = build_embedding_model(input_shape, num_classes=None)
    # 2. Charger les poids depuis le fichier H5 (by_name=True, skip_mismatch=True)
    inference_model.load_weights(h5_path, by_name=True, skip_mismatch=True)
    return inference_model


# -------------------------------------------------------------------
# Création de la galerie d'embeddings (normalisés manuellement)
# -------------------------------------------------------------------
def create_gallery(embedding_model, images_dir, output_path):
    """
    Calcule les embeddings de référence à partir d'un dossier organisé
    en sous-dossiers/personne.
    """
    os.makedirs(output_path, exist_ok=True)
    person_folders = sorted([
        d for d in os.listdir(images_dir)
        if os.path.isdir(os.path.join(images_dir, d))
    ])
    print(f"✓ {len(person_folders)} personnes trouvées")

    embeddings_list = []
    labels_list = []
    label_to_name = {}

    for idx, person_folder in enumerate(person_folders):
        person_path = os.path.join(images_dir, person_folder)
        label_to_name[idx] = person_folder

        img_files = [f for f in os.listdir(person_path)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        print(f"  [{idx+1}/{len(person_folders)}] {person_folder}: {len(img_files)} images", end=' ')
        person_emb_count = 0

        for img_name in img_files:
            img_path = os.path.join(person_path, img_name)
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_resized = cv2.resize(img_rgb, (160, 160))
                img_batch = np.expand_dims(
                    img_resized.astype('float32') / 255.0, axis=0
                )
                # Prédiction (sortie non normalisée)
                emb_raw = embedding_model.predict(img_batch, verbose=0)[0]
                # Normalisation L2 manuelle
                emb = emb_raw / (np.linalg.norm(emb_raw, ord=2) + 1e-10)
                embeddings_list.append(emb)
                labels_list.append(idx)
                person_emb_count += 1
            except Exception as e:
                print(f"\n    ⚠ Erreur {img_name}: {e}", end='')

        print(f"✓ ({person_emb_count} embeddings)")

    embeddings_array = np.array(embeddings_list)
    np.save(os.path.join(output_path, 'embeddings.npy'), embeddings_array)

    mapping = {
        'label_to_name': label_to_name,
        'labels': np.array(labels_list).tolist()
    }
    with open(os.path.join(output_path, 'gallery_info.json'), 'w') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Galerie créée : {len(embeddings_array)} embeddings pour {len(label_to_name)} personnes")
    print(f"   📁 Sauvegardé dans: {output_path}")


# -------------------------------------------------------------------
# Chargement de la galerie
# -------------------------------------------------------------------
def load_gallery(gallery_dir):
    """Charge la galerie et retourne embeddings, labels, label_to_name."""
    embeddings = np.load(os.path.join(gallery_dir, 'embeddings.npy'))
    with open(os.path.join(gallery_dir, 'gallery_info.json'), 'r') as f:
        info = json.load(f)
    label_to_name = {int(k): v for k, v in info['label_to_name'].items()}
    labels = np.array(info['labels'])
    return embeddings, labels, label_to_name


# -------------------------------------------------------------------
# Point d'entrée pour test / création de la galerie
# -------------------------------------------------------------------
if __name__ == "__main__":
    model_path = "C:/PROJETS/Reconnaissance_faciale/Projet_RF/und_train/cores/FineTuning/models/myFace_embedding_model.h5"
    data_path = "C:/PROJETS/RF_dataset/DatasetOriginal_clean_aug"
    gallery_path = "C:/PROJETS/Reconnaissance_faciale/Projet_RF/und_train/cores/FineTuning/gallery"

    print("⏳ Reconstruction du modèle d'inférence (sans Lambda)...")
    embedding_model = build_inference_model_from_weights(model_path)
    print("✅ Modèle chargé avec succès (normalisation L2 manuelle)")

    print("⏳ Création de la galerie...")
    os.makedirs(gallery_path, exist_ok=True)
    create_gallery(embedding_model, data_path, gallery_path)
    print("✅ Galerie créée avec succès!")