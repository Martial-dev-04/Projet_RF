import cv2
import numpy as np
import tensorflow as tf
from utils2 import load_gallery, build_inference_model_from_weights
import os

# Paramètres
MODEL_PATH = '../models/embedding_model.h5'
GALLERY_DIR = '../gallery'
THRESHOLD = 0.6  # cosinus distance : plus petit = plus proche
IMG_SIZE = 160

# Chargement du modèle d'embedding (version sans Lambda)
print("⏳ Chargement du modèle d'inférence...")
embedding_model = build_inference_model_from_weights(MODEL_PATH)

# Pour accélérer, on peut convertir en TFLite (à voir si nécessaire)
print("⏳ Chargement de la galerie...")
embeddings_db, labels_db, label_to_name = load_gallery(GALLERY_DIR)

# Initialiser le détecteur de visage (OpenCV DNN - rapide)
prototxt = "deploy.prototxt"  # à télécharger
caffemodel = "res10_300x300_ssd_iter_140000.caffemodel"  # à télécharger
net = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)

def recognize_face(face_img):
    """Extrait l'embedding et retourne le nom le plus proche."""
    # Prétraitement
    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, (IMG_SIZE, IMG_SIZE))
    face_batch = np.expand_dims(face_resized.astype('float32') / 255.0, axis=0)
    emb_query = embedding_model.predict(face_batch, verbose=0)[0]
    # Distance cosinus (1 - similarité cosinus)
    similarities = np.dot(embeddings_db, emb_query)  # car normalisés, cos = dot
    # On cherche la plus grande similarité (distance minimale)
    max_sim = np.max(similarities)
    if max_sim >= 1 - THRESHOLD:  # seuil converti en similarité
        best_idx = np.argmax(similarities)
        best_label = labels_db[best_idx]
        name = label_to_name[best_label]
        confidence = max_sim
        return name, confidence
    else:
        return "Inconnu", 0.0

# Capture vidéo
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")
            # Extraire le visage
            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue
            name, conf_rec = recognize_face(face_crop)
            text = f"{name} ({conf_rec:.2f})" if name != "Inconnu" else "Inconnu"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, text, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow('Reconnaissance faciale', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()