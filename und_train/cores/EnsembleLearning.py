"""
Workaround pour éviter circular imports
"""
import sys
import importlib

# Vider le cache des modules si nécessaire
if 'EnsembleLearning' in sys.modules:
    del sys.modules['EnsembleLearning']
    
"""======================================================================"""

import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from cores.helpers import HELPERS


class EnsembleRecognizer:
    """
    Reconnaissance faciale avec Ensemble Learning
    Combine multiple extracteurs d'embeddings + multiple classifieurs
    """
    
    def __init__(self, config=None):
        """
        Initialiser l'Ensemble Recognizer
        
        Paramètres:
        - config: dictionnaire avec configuration
            {
                'embedding_models': ['ArcFace', 'VGG-Face', 'Facenet512'],
                'classifiers': ['svm', 'knn', 'rf'],
                'voting_strategy': 'soft'  # 'soft' ou 'hard'
            }
        """
        self.config = config or {
            'embedding_models': ['ArcFace', 'VGG-Face', 'Facenet512'], 
            'classifiers': ['svm', 'knn', 'rf'],
            'voting_strategy': 'soft'
        }
        
        self.classifiers = {}
        self.label_encoder = None
        self.scaler = None
        
        HELPERS.log(f"🤖 EnsembleRecognizer initialisé avec {len(self.config['embedding_models'])} extracteurs", "INFOS")
    
    # ====== SÉPARATION TRAIN/VAL/TEST (À FAIRE AVANT EXTRACTION) ======
    
    def load_image_paths(self, dataset_path):
        """
        Charge les chemins de TOUTES les images organisées par personne
        
        Paramètres:
        - dataset_path: chemin du dataset organisé par personne
        
        Retourne: dict {person_name: [list of image paths]}
        """
        HELPERS.log(f"📂 Chargement des chemins d'images depuis {dataset_path}...", "INFO")
        
        image_paths = {}
        
        for person in os.listdir(dataset_path):
            person_path = os.path.join(dataset_path, person)
            
            if not os.path.isdir(person_path):
                continue
            
            image_paths[person] = []
            
            for img_file in os.listdir(person_path):
                if HELPERS.is_image_file(img_file):
                    img_path = os.path.join(person_path, img_file)
                    image_paths[person].append(img_path)
        
        total_images = sum(len(paths) for paths in image_paths.values())
        HELPERS.log(f"✅ {len(image_paths)} personnes, {total_images} images chargées", "INFO")
        
        return image_paths
    
    def split_and_organize_dataset(self, image_paths, output_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_seed=42):
        """
        ✅ NOUVEAU: Sépare les images ET crée la structure physique train/val/test
        
        Cette méthode:
        1. Sépare les chemins en train/val/test
        2. Crée les dossiers train/val/test
        3. Copie les images dans les dossiers correspondants
        4. Retourne les chemins finaux
        
        ⚠️ À faire APRÈS equilibrate_by_duplication()
        
        Paramètres:
        - image_paths: dict {person_name: [list of image paths]}
        - output_path: chemin de sortie (créera train/ val/ test/ dedans)
        - train_ratio: proportion pour train (défaut: 0.7)
        - val_ratio: proportion pour val (défaut: 0.15)
        - test_ratio: proportion pour test (défaut: 0.15)
        - random_seed: graine pour reproductibilité
        
        Retourne: dict avec les chemins finaux
        {
            'train_path': '/path/train',
            'val_path': '/path/val',
            'test_path': '/path/test',
            'train_paths': dict {person: [paths]},
            'val_paths': dict {person: [paths]},
            'test_paths': dict {person: [paths]}
        }
        """
        import random
        import shutil
        
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        HELPERS.log(f"🔀 SÉPARATION ET ORGANISATION du dataset", "INFO")
        HELPERS.log(f"   Train: {train_ratio*100:.0f}% | Val: {val_ratio*100:.0f}% | Test: {test_ratio*100:.0f}%", "INFO")
        
        # Créer les dossiers principaux
        train_dir = os.path.join(output_path, "train")
        val_dir = os.path.join(output_path, "val")
        test_dir = os.path.join(output_path, "test")
        
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        
        train_paths = {}
        val_paths = {}
        test_paths = {}
        
        # Séparer ET copier pour CHAQUE personne
        for person, paths in image_paths.items():
            # Créer les sous-dossiers pour chaque personne
            train_person_dir = os.path.join(train_dir, person)
            val_person_dir = os.path.join(val_dir, person)
            test_person_dir = os.path.join(test_dir, person)
            
            os.makedirs(train_person_dir, exist_ok=True)
            os.makedirs(val_person_dir, exist_ok=True)
            os.makedirs(test_person_dir, exist_ok=True)
            
            # Mélanger et séparer
            paths_shuffled = paths.copy()
            random.shuffle(paths_shuffled)
            
            n_train = int(len(paths_shuffled) * train_ratio)
            n_val = int(len(paths_shuffled) * val_ratio)
            
            train_person_paths = paths_shuffled[:n_train]
            val_person_paths = paths_shuffled[n_train:n_train + n_val]
            test_person_paths = paths_shuffled[n_train + n_val:]
            
            # Copier les images et stocker les chemins
            train_paths[person] = []
            for img_path in train_person_paths:
                img_name = os.path.basename(img_path)
                output_img_path = os.path.join(train_person_dir, img_name)
                shutil.copy2(img_path, output_img_path)
                train_paths[person].append(output_img_path)
            
            val_paths[person] = []
            for img_path in val_person_paths:
                img_name = os.path.basename(img_path)
                output_img_path = os.path.join(val_person_dir, img_name)
                shutil.copy2(img_path, output_img_path)
                val_paths[person].append(output_img_path)
            
            test_paths[person] = []
            for img_path in test_person_paths:
                img_name = os.path.basename(img_path)
                output_img_path = os.path.join(test_person_dir, img_name)
                shutil.copy2(img_path, output_img_path)
                test_paths[person].append(output_img_path)
            
            HELPERS.log(f"  👤 {person}: {len(train_paths[person])} train | {len(val_paths[person])} val | {len(test_paths[person])} test", "INFO")
        
        train_count = sum(len(v) for v in train_paths.values())
        val_count = sum(len(v) for v in val_paths.values())
        test_count = sum(len(v) for v in test_paths.values())
        
        HELPERS.log(f"✅ Séparation complète:", "INFO")
        HELPERS.log(f"   Train: {train_count} images → {train_dir}", "INFO")
        HELPERS.log(f"   Val:   {val_count} images → {val_dir}", "INFO")
        HELPERS.log(f"   Test:  {test_count} images → {test_dir}", "INFO")
        
        return {
            'train_path': train_dir,
            'val_path': val_dir,
            'test_path': test_dir,
            'train_paths': train_paths,
            'val_paths': val_paths,
            'test_paths': test_paths
        }
    
    
    
    def encode_faces_ensemble_from_paths(self, image_paths_dict, enforce_detection=False, set_name="dataset"):
        """
        Extrait les embeddings à partir d'un dict de chemins d'images
        Utiliser pour train/val/test SÉPARÉMENT après split_dataset_paths()
        
        Paramètres:
        - image_paths_dict: dict {person_name: [list of image paths]}
        - enforce_detection: strict ou permissif pour la détection de visage
        - set_name: nom du dataset (train/val/test) pour les logs
        
        Retourne: (X_ensemble, y)
        - X_ensemble = dict {model_name: embeddings_array}
        - y = labels array (correspondant aux embeddings)
        """
        from deepface import DeepFace
        
        HELPERS.log(f"🔄 Extraction des embeddings ({set_name}) pour {len(self.config['embedding_models'])} modèles...", "INFO")
        
        X_ensemble = {model: [] for model in self.config['embedding_models']}
        y_labels = []
        errors_per_model = {model: 0 for model in self.config['embedding_models']}
        processed_count = 0
        
        # Parcourir les personnes
        for person, paths_list in image_paths_dict.items():
            HELPERS.log(f"👤🪄 {set_name} - Traitement : {person} ({len(paths_list)} images)...", "INFO")
            
            # Parcourir les images de la personne
            for img_path in paths_list:
                img_name = os.path.basename(img_path)
                
                embeddings_extracted = 0
                
                # Extraire avec CHAQUE modèle
                for model_name in self.config['embedding_models']:
                    try:
                        embedding = DeepFace.represent(
                            img_path=img_path,
                            model_name=model_name,
                            detector_backend='skip'
                        )[0]["embedding"]
                        
                        X_ensemble[model_name].append(embedding)
                        embeddings_extracted += 1
                    
                    except Exception as e:
                        errors_per_model[model_name] += 1
                
                # Ajouter le label si au moins 1 embedding a été extrait
                if embeddings_extracted > 0:
                    y_labels.append(person)
                    processed_count += 1
        
        # Convertir en arrays numpy
        X_ensemble_np = {}
        for model_name, embeddings in X_ensemble.items():
            X_ensemble_np[model_name] = np.array(embeddings)
            HELPERS.log(f"  ✅ {model_name}: {len(embeddings)} embeddings", "INFO")
        
        HELPERS.log(f"✅ {set_name} ({processed_count} images traitées)", "INFO")
        
        return X_ensemble_np, np.array(y_labels)
    
    # ====== WORKFLOW COMPLET RECOMMANDÉ (NOUVEAU) ======
    
    def train_val_test_ensemble_complete_v2(self, balanced_dataset_path, output_path, train_ratio=0.7, val_ratio=0.15, 
                                              test_ratio=0.15, random_seed=42, enforce_detection=False):
        """
        ✅ NOUVEAU WORKFLOW RECOMMANDÉ (v2):
        1️⃣ Charger les chemins du dataset équilibré
        2️⃣ Séparer ET organiser en dossiers train/val/test
        3️⃣ Extraire les embeddings SÉPARÉMENT pour chaque ensemble
        4️⃣ Sauvegarder les embeddings dans une structure train/val/test
        
        ⚠️ IMPORTANT: 
        - Le dataset doit déjà être équilibré (voir equilibrate_by_duplication())
        - Les classes sont équilibrées, donc PAS BESOIN d'augmentation
        - Cette méthode crée une structure physique train/val/test
        
        Paramètres:
        - balanced_dataset_path: chemin du dataset équilibré
        - output_path: chemin de sortie (créera dataset_final/ et embeddings/ dedans)
        - train_ratio: proportion d'entraînement (défaut: 0.7)
        - val_ratio: proportion de validation (défaut: 0.15)
        - test_ratio: proportion de test (défaut: 0.15)
        - random_seed: graine aléatoire pour reproductibilité
        - enforce_detection: contrôle strict de la détection de visage
        
        Retourne: dict avec tout ce qui est nécessaire
        {
            'dataset_final_path': '/path/to/dataset_final',
            'embeddings_path': '/path/to/embeddings_ensemble',
            'X_train_ensemble': dict, 'y_train': array,
            'X_val_ensemble': dict, 'y_val': array,
            'X_test_ensemble': dict, 'y_test': array,
            'train_paths': dict, 'val_paths': dict, 'test_paths': dict
        }
        """
        os.makedirs(output_path, exist_ok=True)
        
        HELPERS.log("=" * 60, "INFO")
        HELPERS.log("🚀 NOUVEAU WORKFLOW v2: SÉPARER → EXTRAIRE", "INFO")
        HELPERS.log("=" * 60, "INFO")
        
        # Étape 1: Charger les chemins du dataset équilibré
        image_paths = self.load_image_paths(balanced_dataset_path)
        
        # Étape 2: Séparer ET organiser en dossiers (créer structure physique)
        dataset_final_path = os.path.join(output_path, "dataset_final")
        separation_result = self.split_and_organize_dataset(
            image_paths,
            output_path=dataset_final_path,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            random_seed=random_seed
        )
        
        # Étape 3: Extraire les embeddings SÉPARÉMENT
        X_train_ensemble, y_train = self.encode_faces_ensemble_from_paths(
            separation_result['train_paths'],
            enforce_detection=enforce_detection,
            set_name="TRAIN"
        )
        
        X_val_ensemble, y_val = self.encode_faces_ensemble_from_paths(
            separation_result['val_paths'],
            enforce_detection=enforce_detection,
            set_name="VALIDATION"
        )
        
        X_test_ensemble, y_test = self.encode_faces_ensemble_from_paths(
            separation_result['test_paths'],
            enforce_detection=enforce_detection,
            set_name="TEST"
        )
        
        # Étape 4: Sauvegarder les embeddings dans structure train/val/test
        embeddings_path = os.path.join(output_path, "embeddings_ensemble")
        embeddings_result = self.save_embeddings_structured(
            X_train_ensemble, y_train,
            X_val_ensemble, y_val,
            X_test_ensemble, y_test,
            output_folder=embeddings_path
        )
        
        HELPERS.log("=" * 60, "INFO")
        HELPERS.log("✅ WORKFLOW v2 TERMINÉ", "INFO")
        HELPERS.log("=" * 60, "INFO")
        HELPERS.log(f"📁 Dataset final: {dataset_final_path}", "INFO")
        HELPERS.log(f"📁 Embeddings: {embeddings_path}", "INFO")
        
        return {
            'dataset_final_path': dataset_final_path,
            'embeddings_path': embeddings_path,
            'X_train_ensemble': X_train_ensemble,
            'y_train': y_train,
            'X_val_ensemble': X_val_ensemble,
            'y_val': y_val,
            'X_test_ensemble': X_test_ensemble,
            'y_test': y_test,
            'train_paths': separation_result['train_paths'],
            'val_paths': separation_result['val_paths'],
            'test_paths': separation_result['test_paths'],
            'embeddings_result': embeddings_result
        }
    
    # ====== WORKFLOW COMPLET (ANCIENNE VERSION - OPTIONNEL) ======
    
    
    
    # ====== EXTRACTION D'EMBEDDINGS MULTIPLES (ANCIENNE MÉTHODE - ⚠️ NON RECOMMANDÉE) ======
    
    
    
    def combine_embeddings(self, X_ensemble, strategy='concatenate'):
        """
        Combine les embeddings de plusieurs modèles
        
        Paramètres:
        - X_ensemble: dict {model_name: embeddings}
        - strategy: 'concatenate', 'average', ou 'pca'
        
        Retourne: X_combined (array)
        """
        HELPERS.log(f"🔗 Combination des embeddings (stratégie: {strategy})", "INFO")
        
        if strategy == 'concatenate':
            # Normaliser chaque modèle puis concaténer
            X_combined = []
            for model_name in self.config['embedding_models']:
                embeddings = X_ensemble[model_name]
                # Normaliser L2
                embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
                X_combined.append(embeddings_norm)
            
            X_combined = np.concatenate(X_combined, axis=1)
            HELPERS.log(f"✅ Embeddings combinés : shape {X_combined.shape}", "INFO")
        
        elif strategy == 'average':
            # Moyenne des embeddings normalisés
            embeddings_list = []
            for model_name in self.config['embedding_models']:
                embeddings = X_ensemble[model_name]
                embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
                embeddings_list.append(embeddings_norm)
            
            X_combined = np.mean(embeddings_list, axis=0)
            HELPERS.log(f"✅ Embeddings moyennés : shape {X_combined.shape}", "INFO")
        
        elif strategy == 'pca':
            from sklearn.decomposition import PCA
            # Concaténer puis réduire avec PCA
            X_temp = []
            for model_name in self.config['embedding_models']:
                embeddings = X_ensemble[model_name]
                embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
                X_temp.append(embeddings_norm)
            
            X_concat = np.concatenate(X_temp, axis=1)
            pca = PCA(n_components=0.95)  # Garder 95% de la variance
            X_combined = pca.fit_transform(X_concat)
            self.pca = pca  # Sauvegarder pour prédictions futures
            HELPERS.log(f"✅ PCA appliqué : {X_concat.shape} → {X_combined.shape}", "INFO")
        
        return X_combined
    
    def save_encodings(self, X_ensemble, y, filepath='embeddings_ensemble.pkl'):
        """
        Sauvegarde les embeddings multiples dans un fichier
        
        Paramètres:
        - X_ensemble: dict des embeddings
        - y: labels
        - filepath: chemin du fichier de sortie
        """
        import joblib
        from datetime import datetime
        
        data = {
            'X_ensemble': X_ensemble,
            'y': y,
            'timestamp': datetime.now().isoformat(),
            'config': self.config
        }
        
        joblib.dump(data, filepath)
        HELPERS.log(f"✅ Encodages ensemble sauvegardés : {filepath}", "INFO")
        return filepath
    
    def save_embeddings_structured(self, X_train, y_train, X_val, y_val, X_test, y_test, output_folder='embeddings_ensemble'):
        """
        ✅ NOUVEAU: Sauvegarde les embeddings dans une structure train/val/test
        
        Crée une structure de dossiers:
        embeddings_ensemble/
        ├── train/
        │   ├── X_ensemble.pkl (dict avec tous les modèles)
        │   └── y.pkl
        ├── val/
        │   ├── X_ensemble.pkl
        │   └── y.pkl
        ├── test/
        │   ├── X_ensemble.pkl
        │   └── y.pkl
        └── metadata.json
        
        Paramètres:
        - X_train, y_train: embeddings et labels du train
        - X_val, y_val: embeddings et labels du val
        - X_test, y_test: embeddings et labels du test
        - output_folder: chemin du dossier de sortie
        
        Retourne: dict avec les chemins des fichiers sauvegardés
        """
        import joblib
        import json
        from datetime import datetime
        
        os.makedirs(output_folder, exist_ok=True)
        
        HELPERS.log(f"📦 Sauvegarde des embeddings (structure train/val/test)", "INFO")
        
        # Créer les sous-dossiers
        train_dir = os.path.join(output_folder, "train")
        val_dir = os.path.join(output_folder, "val")
        test_dir = os.path.join(output_folder, "test")
        
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        
        # Sauvegarder TRAIN
        train_x_path = os.path.join(train_dir, "X_ensemble.pkl")
        train_y_path = os.path.join(train_dir, "y.pkl")
        joblib.dump(X_train, train_x_path)
        joblib.dump(y_train, train_y_path)
        HELPERS.log(f"  ✅ Train: {train_x_path}", "INFO")
        
        # Sauvegarder VAL
        val_x_path = os.path.join(val_dir, "X_ensemble.pkl")
        val_y_path = os.path.join(val_dir, "y.pkl")
        joblib.dump(X_val, val_x_path)
        joblib.dump(y_val, val_y_path)
        HELPERS.log(f"  ✅ Val: {val_x_path}", "INFO")
        
        # Sauvegarder TEST
        test_x_path = os.path.join(test_dir, "X_ensemble.pkl")
        test_y_path = os.path.join(test_dir, "y.pkl")
        joblib.dump(X_test, test_x_path)
        joblib.dump(y_test, test_y_path)
        HELPERS.log(f"  ✅ Test: {test_x_path}", "INFO")
        
        # Sauvegarder les métadonnées
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'train': {
                'X_ensemble_path': train_x_path,
                'y_path': train_y_path,
                'num_samples': len(y_train),
                'num_models': len(X_train)
            },
            'val': {
                'X_ensemble_path': val_x_path,
                'y_path': val_y_path,
                'num_samples': len(y_val),
                'num_models': len(X_val)
            },
            'test': {
                'X_ensemble_path': test_x_path,
                'y_path': test_y_path,
                'num_samples': len(y_test),
                'num_models': len(X_test)
            }
        }
        
        metadata_path = os.path.join(output_folder, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        HELPERS.log(f"✅ Embeddings sauvegardés dans: {output_folder}", "INFO")
        HELPERS.log(f"   Train: {len(y_train)} samples", "INFO")
        HELPERS.log(f"   Val:   {len(y_val)} samples", "INFO")
        HELPERS.log(f"   Test:  {len(y_test)} samples", "INFO")
        
        return {
            'folder': output_folder,
            'train_dir': train_dir,
            'val_dir': val_dir,
            'test_dir': test_dir,
            'metadata_path': metadata_path
        }
    
    def load_embeddings_structured(self, embeddings_folder):
        """
        ✅ NOUVEAU: Charge les embeddings depuis la structure train/val/test
        
        Paramètres:
        - embeddings_folder: chemin du dossier contenant train/val/test
        
        Retourne: dict avec tous les embeddings et labels
        {
            'X_train': dict, 'y_train': array,
            'X_val': dict, 'y_val': array,
            'X_test': dict, 'y_test': array,
            'metadata': dict
        }
        """
        import joblib
        import json
        
        HELPERS.log(f"📦 Chargement des embeddings depuis: {embeddings_folder}", "INFO")
        
        # Charger les métadonnées
        metadata_path = os.path.join(embeddings_folder, "metadata.json")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Charger TRAIN
        train_x = joblib.load(os.path.join(embeddings_folder, "train", "X_ensemble.pkl"))
        train_y = joblib.load(os.path.join(embeddings_folder, "train", "y.pkl"))
        
        # Charger VAL
        val_x = joblib.load(os.path.join(embeddings_folder, "val", "X_ensemble.pkl"))
        val_y = joblib.load(os.path.join(embeddings_folder, "val", "y.pkl"))
        
        # Charger TEST
        test_x = joblib.load(os.path.join(embeddings_folder, "test", "X_ensemble.pkl"))
        test_y = joblib.load(os.path.join(embeddings_folder, "test", "y.pkl"))
        
        HELPERS.log(f"✅ Embeddings chargés:", "INFO")
        HELPERS.log(f"   Train: {len(train_y)} samples", "INFO")
        HELPERS.log(f"   Val:   {len(val_y)} samples", "INFO")
        HELPERS.log(f"   Test:  {len(test_y)} samples", "INFO")
        
        return {
            'X_train': train_x,
            'y_train': train_y,
            'X_val': val_x,
            'y_val': val_y,
            'X_test': test_x,
            'y_test': test_y,
            'metadata': metadata
        }
    
    def load_encodings(self, filepath='embeddings_ensemble.pkl'):
        """
        Charge les embeddings multiples depuis un fichier
        
        Paramètres:
        - filepath: chemin du fichier d'entrée
        
        Retourne: (X_ensemble, y)
        """
        import joblib
        
        if not os.path.exists(filepath):
            HELPERS.log(f"❌ Fichier non trouvé : {filepath}", "ERROR")
            return None, None
        
        data = joblib.load(filepath)
        X_ensemble = data['X_ensemble']
        y = data['y']
        
        HELPERS.log(f"✅ Encodages ensemble chargés : {len(y)} labels", "INFO")
        return X_ensemble, y
    
    # ====== ENTRAÎNEMENT ENSEMBLE DE CLASSIFIEURS ======
    
    def train_ensemble(self, X_train, y_train):
        """
        Entraîne plusieurs classifieurs sur les embeddings
        
        Paramètres:
        - X_train: features d'entraînement
        - y_train: labels d'entraînement
        """
        from sklearn.svm import SVC
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder, StandardScaler
        
        HELPERS.log(f"🚀 Entraînement de {len(self.config['classifiers'])} classifieurs...", "INFO")
        
        # Encoder les labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y_train)
        
        # Normaliser les features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_train)
        
        # Entraîner chaque classifieur
        if 'svm' in self.config['classifiers']:
            HELPERS.log("  → Entraînement SVM (RBF)...", "INFO")
            self.classifiers['svm'] = SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=42
            )
            self.classifiers['svm'].fit(X_scaled, y_encoded)
        
        if 'knn' in self.config['classifiers']:
            HELPERS.log("  → Entraînement KNN (k=5)...", "INFO")
            self.classifiers['knn'] = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
            self.classifiers['knn'].fit(X_scaled, y_encoded)
        
        if 'rf' in self.config['classifiers']:
            HELPERS.log("  → Entraînement Random Forest (100 trees)...", "INFO")
            self.classifiers['rf'] = RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                random_state=42,
                n_jobs=-1
            )
            self.classifiers['rf'].fit(X_scaled, y_encoded)
        
        HELPERS.log(f"✅ {len(self.classifiers)} classifieurs entraînés", "INFO")
        
        return self.classifiers
    
    # ====== PRÉDICTION AVEC VOTATION ======
    
    def predict_ensemble(self, X_test):
        """
        Prédit avec votation de plusieurs classifieurs
        
        Paramètres:
        - X_test: features de test
        
        Retourne: (predictions, confidences)
        """
        
        # Normaliser
        X_scaled = self.scaler.transform(X_test)
        
        if self.config['voting_strategy'] == 'hard':
            # Votation dure : majorité simple
            predictions_list = []
            for clf_name, clf in self.classifiers.items():
                preds = clf.predict(X_scaled)
                predictions_list.append(preds)
            
            # Voter
            predictions = []
            confidences = []
            for i in range(len(X_test)):
                votes = [preds[i] for preds in predictions_list]
                # Prendre le plus fréquent
                most_common = max(set(votes), key=votes.count)
                predictions.append(most_common)
                # Confiance = proportion de votes favorables
                confidence = votes.count(most_common) / len(self.classifiers)
                confidences.append(confidence)
        
        elif self.config['voting_strategy'] == 'soft':
            # Votation souple : moyenne des probabilités
            proba_list = []
            for clf_name, clf in self.classifiers.items():
                proba = clf.predict_proba(X_scaled)
                proba_list.append(proba)
            
            # Moyenne des probabilités
            mean_proba = np.mean(proba_list, axis=0)
            predictions = np.argmax(mean_proba, axis=1)
            confidences = np.max(mean_proba, axis=1)
        
        return predictions, np.array(confidences)
    
    def predict_single(self, embedding, confidence_threshold=0.85):
        """
        Prédit pour un SEUL embedding
        
        Paramètres:
        - embedding: embedding combiné (déjà combiné des 3 modèles)
        - confidence_threshold: seuil de confiance minimum
        
        Retourne: dict avec 'name', 'confidence', 'is_confident'
        """
        pred_label, confidence = self.predict_ensemble(np.array([embedding]).reshape(1, -1))
        pred_label = pred_label[0]
        confidence = confidence[0]
        
        name = self.label_encoder.inverse_transform([pred_label])[0]
        is_confident = confidence >= confidence_threshold
        
        return {
            'name': name,
            'confidence': confidence,
            'is_confident': is_confident
        }
    
    # ====== ÉVALUATION ======
    
    def evaluate_ensemble(self, X_test, y_test):
        """
        Évalue les performances du modèle ensemble
        
        Paramètres:
        - X_test: features de test
        - y_test: labels de test
        
        Retourne: dict avec métriques
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        HELPERS.log("📊 Évaluation de l'Ensemble...", "INFO")
        
        y_encoded = self.label_encoder.transform(y_test)
        predictions, confidences = self.predict_ensemble(X_test)
        
        # Métriques
        accuracy = accuracy_score(y_encoded, predictions)
        precision = precision_score(y_encoded, predictions, average='weighted', zero_division=0)
        recall = recall_score(y_encoded, predictions, average='weighted', zero_division=0)
        f1 = f1_score(y_encoded, predictions, average='weighted', zero_division=0)
        
        HELPERS.log(f"✅ Accuracy:  {accuracy:.4f}", "INFO")
        HELPERS.log(f"   Precision: {precision:.4f}", "INFO")
        HELPERS.log(f"   Recall:    {recall:.4f}", "INFO")
        HELPERS.log(f"   F1-Score:  {f1:.4f}", "INFO")
        
        mean_confidence = np.mean(confidences)
        HELPERS.log(f"   Confiance moyenne: {mean_confidence:.4f}", "INFO")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'mean_confidence': mean_confidence,
            'predictions': predictions,
            'confidences': confidences
        }
    
    def compare_classifiers(self, X_test, y_test):
        """
        Compare les performances de chaque classifieur individuellement
        
        Paramètres:
        - X_test: features de test
        - y_test: labels de test
        
        Retourne: dict {classifier_name: accuracy}
        """
        from sklearn.metrics import accuracy_score
        
        HELPERS.log("🔍 Comparaison des classifieurs individuels...", "INFO")
        
        y_encoded = self.label_encoder.transform(y_test)
        X_scaled = self.scaler.transform(X_test)
        
        results = {}
        for clf_name, clf in self.classifiers.items():
            preds = clf.predict(X_scaled)
            acc = accuracy_score(y_encoded, preds)
            results[clf_name] = acc
            HELPERS.log(f"  {clf_name.upper()}: {acc:.4f}", "INFO")
        
        return results
    
    def confusion_matrix_ensemble(self, X_test, y_test):
        """
        Visualise la matrice de confusion du modèle ensemble
        
        Paramètres:
        - X_test: features de test
        - y_test: labels de test
        """
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        
        HELPERS.log("📉 Génération matrice de confusion...", "INFO")
        
        y_encoded = self.label_encoder.transform(y_test)
        predictions, _ = self.predict_ensemble(X_test)
        
        matrix = confusion_matrix(y_encoded, predictions)
        
        # Visualiser
        plt.figure(figsize=(14, 12))
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.label_encoder.classes_,
                    yticklabels=self.label_encoder.classes_)
        plt.ylabel('Vrai label')
        plt.xlabel('Label prédit')
        plt.title('Matrice de Confusion - Ensemble Learning')
        plt.tight_layout()
        plt.show()
        
        return matrix
    
    # ====== SAUVEGARDE/CHARGEMENT ======
    
    def save_ensemble(self, filepath='C:/PROJETS/Reconnaissance_faciale/Projet_RF/modelensemble_model.pkl'):
        """
        Sauvegarde tout le modèle ensemble
        
        Paramètres:
        - filepath: chemin du fichier de sortie
        """
        import joblib
        from datetime import datetime
        
        data = {
            'classifiers': self.classifiers,
            'label_encoder': self.label_encoder,
            'scaler': self.scaler,
            'config': self.config,
            'pca': getattr(self, 'pca', None),
            'timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(data, filepath)
        HELPERS.log(f"✅ Modèle ensemble sauvegardé : {filepath}", "INFO")
        return filepath
    
    def load_ensemble(self, filepath='ensemble_model.pkl'):
        """
        Charge un modèle ensemble sauvegardé
        
        Paramètres:
        - filepath: chemin du fichier d'entrée
        """
        import joblib
        
        if not os.path.exists(filepath):
            HELPERS.log(f"❌ Fichier non trouvé : {filepath}", "ERROR")
            return False
        
        data = joblib.load(filepath)
        self.classifiers = data['classifiers']
        self.label_encoder = data['label_encoder']
        self.scaler = data['scaler']
        self.config = data['config']
        if data.get('pca'):
            self.pca = data['pca']
        
        HELPERS.log(f"✅ Modèle ensemble chargé : {filepath}", "INFO")
        return True