import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2

def build_embedding_model(input_shape=(160, 160, 3), num_classes=None):
    """Construit MobileNetV2 + projection embedding 128 + option tête softmax."""
    base_model = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    base_model.trainable = False

    inputs = keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    
    # ✅ Ajouter BatchNorm + Dropout
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    # Embedding avec L2 regularization
    x = layers.Dense(
        128, 
        activation=None,
        kernel_regularizer=keras.regularizers.l2(1e-4),  # ✅ L2 reg
        name='embedding'
    )(x)
    
    x = layers.BatchNormalization()(x)
    
    # ✅ FIX: Ajouter output_shape pour la Lambda layer
    embeddings = layers.Lambda(
    lambda v: tf.math.l2_normalize(v, axis=1),
    output_shape=lambda input_shape: input_shape,   # <-- on indique explicitement la forme
    name='l2_norm'
    )(x)

    if num_classes is not None:
        x = layers.Dropout(0.4)(embeddings)  # ✅ Dropout avant classifier
        classifier = layers.Dense(
            num_classes, 
            activation='softmax',
            kernel_regularizer=keras.regularizers.l2(1e-4),  # ✅ L2 reg
            name='classifier'
        )(x)
        model = keras.Model(inputs, classifier)
    else:
        model = keras.Model(inputs, embeddings)

    return model

def create_gallery(embedding_model, images_dir, output_path):
    """Calcule les embeddings de référence à partir d'un dossier organisé en sous-dossiers/personne.
    Structure attendue : images_dir/
                               person_A/ img1.jpg, img2.jpg ...
                               person_B/ ...
    """
    embeddings_list = []
    labels_list = []
    label_to_name = {}
    
    # Créer le dossier de sortie
    os.makedirs(output_path, exist_ok=True)

    # Obtenir la liste des dossiers de personnes
    person_folders = sorted([d for d in os.listdir(images_dir) 
                            if os.path.isdir(os.path.join(images_dir, d))])
    print(f"✓ {len(person_folders)} personnes trouvées")

    for idx, person_folder in enumerate(person_folders):
        person_path = os.path.join(images_dir, person_folder)
        label_to_name[idx] = person_folder
        
        # Obtenir les images
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
                img_batch = np.expand_dims(img_resized.astype('float32') / 255.0, axis=0)
                emb = embedding_model.predict(img_batch, verbose=0)[0]
                embeddings_list.append(emb)
                labels_list.append(idx)
                person_emb_count += 1
            except Exception as e:
                print(f"\n    ⚠ Erreur {img_name}: {e}", end='')
        
        print(f"✓ ({person_emb_count} embeddings)")

    embeddings_array = np.array(embeddings_list)
    
    # Sauvegarder les embeddings
    np.save(os.path.join(output_path, 'embeddings.npy'), embeddings_array)
    
    # Sauvegarder le mapping label->nom et du label de chaque embedding
    mapping = {
        'label_to_name': label_to_name,
        'labels': np.array(labels_list).tolist()  # pour pouvoir recharger
    }
    with open(os.path.join(output_path, 'gallery_info.json'), 'w') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Galerie créée : {len(embeddings_array)} embeddings pour {len(label_to_name)} personnes")
    print(f"   📁 Sauvegardé dans: {output_path}")

def load_gallery(gallery_dir):
    """Charge la galerie et retourne embeddings, labels, label_to_name."""
    embeddings = np.load(os.path.join(gallery_dir, 'embeddings.npy'))
    with open(os.path.join(gallery_dir, 'gallery_info.json'), 'r') as f:
        info = json.load(f)
    label_to_name = {int(k): v for k, v in info['label_to_name'].items()}
    labels = np.array(info['labels'])
    return embeddings, labels, label_to_name

if __name__ == "__main__":
    # Test rapide
    model_path = "C:/PROJETS/Reconnaissance_faciale/Projet_RF/und_train/cores/FineTuning/models/myFace_embedding_model.h5" 
    data_path = "C:/PROJETS/RF_dataset/DatasetOriginal_clean_aug"
    gallery_path = "C:/PROJETS/Reconnaissance_faciale/Projet_RF/und_train/cores/FineTuning/gallery"
    
    # ✅ FIX: Charger avec safe_mode=False pour les Lambda layers
    print("⏳ Chargement du modèle...")
    from tensorflow.keras.layers import Lambda

    # Redéfinit la couche Lambda avec output_shape
    l2_norm_layer = Lambda(
        lambda v: tf.math.l2_normalize(v, axis=1),
        output_shape=lambda input_shape: input_shape,
        name='l2_norm'
    )

    embedding_model = keras.models.load_model(
        model_path,
        custom_objects={'l2_norm': l2_norm_layer},
        safe_mode=False, 
        compile = False
    )
    for layer in embedding_model.layers:
        print(layer.name)
    print("✅ Modèle chargé avec succès")
    
    print("⏳ Création de la galerie...")
    os.makedirs(gallery_path, exist_ok=True)
    create_gallery(embedding_model, data_path, gallery_path)
    print("✅ Galerie créée avec succès!")