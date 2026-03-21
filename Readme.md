# 📚 Système de Reconnaissance Faciale pour la gestion de présence automatisée

Projet complet de gestion de présence par reconnaissance faciale, utilisant DeepFace, SVM, OpenCV et une interface web Flask.  
Les présences sont enregistrées dans un fichier Excel.

---

## 📝 Sommaire

1. [Contexte et objectifs](#contexte-et-objectifs)  
2. [Architecture du projet](#architecture-du-projet)  
3. [Installation et configuration](#installation-et-configuration)  
4. [Pipeline de traitement et d’entraînement](#pipeline-de-traitement-et-dentraînement)  
5. [Utilisation de l’application](#utilisation-de-lapplication)  
6. [Structure des dossiers et fichiers](#structure-des-dossiers-et-fichiers)  
7. [FAQ et dépannage](#faq-et-dépannage)  
8. [Améliorations possibles](#améliorations-possibles)  
9. [Auteur](#auteur)  

---

## 1. Contexte et objectifs

Ce projet vise à automatiser la gestion des présences lors d’événements ou de cours grâce à la reconnaissance faciale.  
Il permet :
- D’identifier automatiquement les personnes via webcam.
- D’enregistrer leur présence dans un fichier Excel.
- De fournir une interface web simple pour visualiser et télécharger les présences.

---

## 2. Architecture du projet

```
[Webcam] 
   ↓
[Détection de visage (face_recognition)]
   ↓
[Extraction d'embeddings (DeepFace)]
   ↓
[Classification (SVM scikit-learn)]
   ↓
[Enregistrement Excel (openpyxl)]
   ↓
[Interface Web (Flask)]
```

---

## 3. Installation et configuration

### a. Prérequis

- Python 3.8 ou supérieur
- Une webcam fonctionnelle

### b. Installation

1. **Cloner le projet**  
   Copie le dossier sur ta machine.

2. **Créer et activer un environnement virtuel**  
   (Optionnel, recommandé)
   ```bash
   python -m venv env_deepface
   env_deepface\Scripts\activate
   ```

3. **Installer les dépendances**
   ```bash
   pip install -r requirements.txt
   ```

---

## 4. Pipeline de traitement et d’entraînement

### a. Préparation et nettoyage du dataset

- Utilise le notebook [`und_dataset.ipynb`](und_dataset.ipynb) pour :
  - Nettoyer les images (suppression des images corrompues, floues, etc.).
  - Normaliser la luminosité et la netteté.
  - Augmenter le dataset (rotation, zoom, contraste…).
  - Redimensionner toutes les images à 224x224 pixels.
  - Organiser les images par classe (un dossier par personne).

### b. Division du dataset

- Séparation en trois ensembles :  
  - **Train** (80%)  
  - **Validation** (10%)  
  - **Test** (10%)  
- Cette étape est automatisée dans le notebook.

### c. Extraction des embeddings et entraînement du modèle

- Utilise le notebook [`train_test.ipynb`](train_test.ipynb) pour :
  - Extraire les embeddings faciaux de chaque image avec DeepFace.
  - Entraîner un SVM sur ces embeddings.
  - Sauvegarder le modèle, l’encodeur de labels et le backbone DeepFace dans `face_model.pkl`.

**Exemple de code d’entraînement :**
```python
from deepface import DeepFace
from sklearn.svm import SVC
import joblib

# Extraction des embeddings
embeddings = []
labels = []
for img_path, label in dataset:
    embedding = DeepFace.represent(img_path, model_name="Facenet")[0]["embedding"]
    embeddings.append(embedding)
    labels.append(label)

# Entraînement SVM
svm = SVC(probability=True)
svm.fit(embeddings, labels)

# Sauvegarde
joblib.dump({"model": svm, "deepface_model": "Facenet"}, "face_model.pkl")
```

---

## 5. Utilisation de l’application

### a. Lancer l’interface web

1. Place `face_model.pkl` et (optionnel) `Liste_de_présence.xlsx` dans le dossier du projet.
2. Va dans le dossier `Interface_RF` et lance :
   ```bash
   python main.py
   ```
3. Ouvre [http://localhost:5000](http://localhost:5000) dans ton navigateur.

### b. Fonctionnalités de l’interface

- **/camera** : Lance la webcam et effectue la reconnaissance faciale en temps réel.
- **/presence** : Affiche la liste des présences du jour.
- **/telecharger_feuille?feuille=JJ-MM-AAAA** : Télécharge la feuille Excel d’un jour précis.
- **/telecharger_excel** : Télécharge l’historique complet.

### c. Enregistrement des présences

- Lorsqu’un visage est reconnu, le nom, l’heure et la date sont ajoutés à la feuille Excel.
- Un même individu n’est enregistré qu’une seule fois par jour.

---

## 6. Structure des dossiers et fichiers

```
Projet_RF/
├── face_model.pkl                # Modèle SVM + backbone DeepFace
├── Liste_de_présence.xlsx        # Historique des présences (Excel)
├── requirements.txt              # Dépendances Python
├── train_test.ipynb              # Notebook d'entraînement et de test du modèle
├── und_dataset.ipynb             # Notebook de préparation/cleaning du dataset
├── img/                          # Images pour le test manuel 
│   ├── img_001.jpg
│   └── img_002.jpg
├── Interface_RF/
│   ├── main.py                   # Application Flask (interface web)
│   ├── static/                   # Fichiers statiques (CSS, images)
│   └── templates/                # Templates HTML Flask
└── env_deepface/                 # Environnement virtuel Python (optionnel)
```

---

## 7. FAQ et dépannage

- **Erreur “No module named ...”**  
  → Installe les dépendances avec `pip install -r requirements.txt`.

- **Webcam non détectée**  
  → Vérifie qu’aucune autre application n’utilise la webcam.

- **Présence non enregistrée**  
  → Vérifie la qualité du dataset et le seuil de confiance dans le code (`confidence >= 0.90`).

- **Problème d’encodage Excel**  
  → Assure-toi que le fichier n’est pas ouvert dans Excel lors de l’écriture.

---

## 8. Améliorations possibles

- Ajout d’une base de données SQL pour les présences.
- Gestion multi-utilisateur et rôles.
- Ajout de notifications ou d’un dashboard statistique.

---

## 9. Auteur

Projet réalisé par :

  → *OLAOGOU Carlos Martial*
  → *BOTON Désiré*
  → *ZINSOU Homas*
 **Membres du Club IA du CAEB/Bénin Excellence/Fondation VALLET de Natitingou.**


---