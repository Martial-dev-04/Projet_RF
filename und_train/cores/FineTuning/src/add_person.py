
import cv2
import numpy as np
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils2 import load_gallery, build_inference_model_from_weights
import tensorflow as tf

MODEL_PATH = '../models/embedding_model.h5'
GALLERY_DIR = '../gallery'
CAPTURE_COUNT = 5  # Nombre d'images à prendre via webcam
IMG_SIZE = 160

def update_gallery(name, embedding_list):
    """Ajoute les embeddings à la galerie existante et sauvegarde."""
    embeddings, labels, label_to_name = load_gallery(GALLERY_DIR)
    
    # Nouveau label
    new_label = max(label_to_name.keys()) + 1 if label_to_name else 0
    label_to_name[new_label] = name
    
    new_embeddings = np.array(embedding_list)
    new_labels = np.full(len(embedding_list), new_label)

    # Concatène
    if len(embeddings) > 0:
        all_embeddings = np.vstack([embeddings, new_embeddings])
        all_labels = np.concatenate([labels, new_labels])
    else:
        all_embeddings = new_embeddings
        all_labels = new_labels

    np.save(os.path.join(GALLERY_DIR, 'embeddings.npy'), all_embeddings)
    with open(os.path.join(GALLERY_DIR, 'gallery_info.json'), 'w') as f:
        json.dump({
            'label_to_name': {int(k): v for k, v in label_to_name.items()},
            'labels': all_labels.tolist()
        }, f)
    print(f"✅Galerie mise à jour : {name} ajouté avec {len(embedding_list)} embeddings.")

def capture_embeddings(embedding_model):
    cap = cv2.VideoCapture(0)
    collected = 0
    embeddings_captured = []
    print(f"Capturez le visage de la nouvelle personne. Appuyez sur ESPACE pour prendre une photo ({CAPTURE_COUNT} nécessaires).")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Ajout personne - ESPACE pour capturer, q pour quitter', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            # Supposons que le visage est bien centré (à la main)
            # Dans une version réelle, on peut intégrer un détecteur ici.
            face_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_resized = cv2.resize(face_rgb, (IMG_SIZE, IMG_SIZE))
            face_batch = np.expand_dims(face_resized.astype('float32') / 255.0, axis=0)
            emb_raw = embedding_model.predict(face_batch, verbose=0)[0]
            
            # Normalisation L2 manuelle
            emb = emb_raw / (np.linalg.norm(emb_raw, ord=2) + 1e-10)
            embeddings_captured.append(emb)
            collected += 1
            print(f"Capture {collected}/{CAPTURE_COUNT}")
        elif key == ord('q'):
            break
        if collected >= CAPTURE_COUNT:
            break
    cap.release()
    cv2.destroyAllWindows()
    return embeddings_captured

if __name__ == '__main__':
    name = input("Nom de la nouvelle personne : ")
    embedding_model = build_inference_model_from_weights(MODEL_PATH)
    new_embs = capture_embeddings(embedding_model)
    if len(new_embs) == CAPTURE_COUNT:
        update_gallery(name, new_embs)
    else:
        print("Moins d'images capturées, opération annulée.")