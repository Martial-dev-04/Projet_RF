"""
Workaround pour éviter circular imports
"""
import sys
import importlib

# Vider le cache des modules si nécessaire
if 'utils' in sys.modules:
    del sys.modules['utils']

from PIL import Image
import cv2
import os
from matplotlib import image
import numpy as np
import random
import matplotlib.pyplot as plt
import face_recognition
from cores.helpers import HELPERS

random.seed(42)

config = {}

class DatasetProcessor:
    def __init__(self, config=0):
        self.config = config
        self.log = []
    
    # ANALYSE DATASET
    
    def get_dataset_stats(self,path):
        """Retourne : nb images, résolutions, distribution classes, etc."""
        stats = {
            "total_images": 0,
            "total_classe": 0,
            "per_class": {},
            "image_sizes": [],
            "brightness_dist": []
            }
        folders = self.read_folder(path) # récupère un tuple de (nombre_sous_dossier, liste_chemin_sous_dossier)
        path_folders = folders[1] # stock liste_chemin_sous_dossier
        
        stats["total_classe"] = folders[0] # stock nombre_sous_dossier
        
        for path_folder in path_folders:                # parcour la liste_chemin_sous_dossier
            fichiers = HELPERS.read_folder(path_folder)    # récupère un tuple de (nombre_fichiers, liste_chemin_fichier)
            
            stats["total_images"] += fichiers[0]        # (incrémente) ajoute nombre_fichiers 
            
            per_class = stats["per_class"]
            per_class[os.path.basename(path_folder)] = fichiers[0]  # ajoute le paire clé : valeur (Nom_dossier : nombre_fichier) au dictionnaire "per_classe"
            
        return stats
    
    
    def augmenter_img(self, image, num_variants=3):
        """
        Génère PLUSIEURS variantes d'une image
        Au lieu d'une seule transformation
        """
        
        transformations = []
    
        for _ in range(num_variants):
            choix = random.choice(['rotation', 'contraste', 'zoom', 'flip', 'noise'])
            
            try:
                if choix == 'rotation':
                    angle = random.uniform(-15, 15)
                    h, w = image.shape[:2]
                    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
                    result = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
                    
                elif choix == 'contraste':
                    alpha = random.uniform(1.1, 1.6)
                    beta = random.randint(-20, 20)
                    result = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
                    
                elif choix == 'zoom':
                    h, w = image.shape[:2]
                    zoom_factor = random.uniform(1.05, 1.4)
                    nh, nw = int(h/zoom_factor), int(w/zoom_factor)
                    y_start = (h - nh) // 2
                    x_start = (w - nw) // 2
                    cropped = image[y_start:y_start+nh, x_start:x_start+nw]
                    result = cv2.resize(cropped, (w, h))
                    
                elif choix == 'flip':
                    result = cv2.flip(image, random.choice([-1, 0, 1]))  # -1=both, 0=vertical, 1=horizontal
                    
                elif choix == 'noise':
                    noise = np.random.normal(0, 5, image.shape)
                    result = cv2.add(image.astype(float), noise)
                    result = np.clip(result, 0, 255).astype(np.uint8)
                
                if result is not None:
                    transformations.append(result)
                    
            except Exception as e:
                HELPERS.log(f"Erreur augmentation ({choix}): {e}", "WARNING")
                continue
        
        return transformations if transformations else [image]
        
    
    def view_image(self, folder_path):
        """Choisir une image au hazard dans le dossier et l'affiche"""

        folder_person = HELPERS.read_folder(folder_path)[1]
        images_exemples = []
        el = random.choice(folder_person)
        
        if HELPERS.is_image_file(el):
            nom = os.path.basename(folder_path)
            images_exemples.append((nom, random.choice(folder_person)))
            
            lines = 1
            columns = 1
            figsize = (20, 20)
            
        else:
            for folder in folder_person:
                fichiers = self.read_folder(folder)[1]
                nom = os.path.basename(folder)
                images_exemples.append((nom, random.choice(fichiers)))
                
            lines = len(images_exemples)//2
            columns = len(images_exemples)//4
            figsize = (30, 30)                 

            

        # Afficher les images d'exemple
        if images_exemples:
            fig, axes = plt.subplots(lines, columns)

            axes = np.array(axes).reshape(-1)  # force en tableau 1D

            for i, (nom, path) in enumerate(images_exemples):
                img = Image.open(path)
                axes[i].imshow(img)
                axes[i].set_title(nom, fontsize=8)
                axes[i].axis('off')
            plt.tight_layout()
            plt.show()
        
        return folder_person
        
    
    def brightness_distribution(self, file_path):
        """Affiche la distribution de la luminosité"""
        name  = os.path.basename(file_path)
        brightness = []
        el = random.choice(HELPERS.read_folder(file_path)[1])
        
        if HELPERS.is_image_file(el):
            print(f"🌕 Calcule des luminosités moyennes pour {name}...")
            fichiers = HELPERS.read_folder(file_path)[1]
            for fichier in fichiers:
                if self.validation_image(fichier):
                    image = cv2.imread(fichier)
                    lum = HELPERS.get_brightness(image)
                    brightness.append(lum)
        else:
            print(f"🌕 Calcule des luminosités moyennes pour {name} (Dataset)...")
            name  = os.path.basename(file_path)
            folders = HELPERS.read_folder(file_path)[1]
            for folder in folders:
                fichiers = HELPERS.read_folder(folder)[1]
                for fichier in fichiers:
                    if self.validation_image(fichier):
                        image = cv2.imread(fichier)
                        lum = HELPERS.get_brightness(image)
                        brightness.append(lum)
        
        
        # Afficher du résultats sous forme d'histogramme
        plt.figure(figsize=(10, 5))
        plt.hist(brightness, bins=20, color='dodgerblue')
        plt.title(f'Distribution de la luminosité moyenne des images pour {name}')
        plt.xlabel('Luminosité moyenne')
        plt.ylabel("Nombre d'images")
        plt.grid(True)
        plt.show()

        # Quelques statistiques
        print(f"Min : {np.min(brightness):.2f} | Max : {np.max(brightness):.2f}") 
                        
        return f"Nom: {name} \nLuminosité moyenne générale : {np.mean(brightness):.2f}"
    
    def sharpness_distribution(self, file_path):
        """Affiche la distribution de la nettété"""
        name  = os.path.basename(file_path)
        sharpness = []
        el = random.choice(HELPERS.read_folder(file_path)[1])
        
        if HELPERS.is_image_file(el):
            print(f"🌕 Calcule de la netétté des images pour {name}...")
            fichiers = HELPERS.read_folder(file_path)[1]
            for fichier in fichiers:
                if self.validation_image(fichier):
                    image = cv2.imread(fichier)
                    lum = HELPERS.get_sharpness(image)
                    sharpness.append(lum)
        else:
            print(f"🌕 Calcule de la netétté des images pour {name} (Dataset)...")
            name  = os.path.basename(file_path)
            folders = HELPERS.read_folder(file_path)[1]
            for folder in folders:
                fichiers = HELPERS.read_folder(folder)[1]
                for fichier in fichiers:
                    if self.validation_image(fichier):
                        image = cv2.imread(fichier)
                        lum = HELPERS.get_sharpness(image)
                        sharpness.append(lum)
    
        # Tracer l'histogramme
        plt.figure(figsize=(10,5))
        plt.hist(sharpness, bins=40, color='orange', edgecolor = 'black')
        plt.title(f"Distribution de la netteté des images normalisée(Variance du Laplacien) pour {name}")
        plt.xlabel("Netteté (variance du Laplacien)")

        # 🧠 Personnalisation des graduations :
        #plt.xticks(ticks=range(0, 2000, 50))  # ici, de 0 à 3000 avec un pas de 200

        plt.ylabel("Nombre d'images")
        plt.axvline(x=50, color='red', linestyle='--', label='Seuil flou(50)' )
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        return  f"Nom: {name} \nNetétté moyenne générale : {np.mean(sharpness):.2f}"
    
    
    # PRÉTRAITEMENT
    
    def process_brightness(self, image, threshold_low=80, threshold_high=170):
        """
        Corrige la luminosité intelligemment
        Utilise CLAHE pour meilleur contraste
        """
        brightness = HELPERS.get_brightness(image)

        # Décider de la stratégie d'ajustement en fonction de la luminosité
        if brightness < threshold_low:
            # Image trop sombre : correction gamma pour éclaircir
            gamma = 1.5
            inv_gamma = 1.0/ gamma
            table = np.array([(i / 255.0) **  inv_gamma * 255 for i in range(256)]).astype("uint8")
            adjusted = cv2.LUT(image, table)
            
        elif brightness > threshold_high:
            # Image trop lumineuse : réduire
            gamma = 0.7
            inv_gamma = 1.0 / gamma
            table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
            adjusted = cv2.LUT(image, table)
            
        else: 
            adjusted = image.copy()  # Pas de correction nécessaire, mais on retourne une copie pour éviter les modifications en place
        
        # Appliquer CLAHE pour améliorer le contraste de manière intelligente
        gray = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Convertir back en couleur
        adjusted = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        
        return adjusted
    
    def process_sharpness(self, image, threshold_low, threshold_high):
        """Corrige la nettété (flou) d'une image"""
        sharpness = HELPERS.get_sharpness(image)
        
        if sharpness < threshold_low:
            # Appliquer un filtre de netteté
            kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]]) 
            adjusted = cv2.filter2D(image, -1, kernel)
            return adjusted
        
        elif sharpness > threshold_high:
            # Appliquer un léger flou pour éviter les artefacts
            adjusted = cv2.GaussianBlur(image, (3, 3), 0)
            return adjusted
        
        return image
    
    # VALIDATION & NETTOYAGE
    
    def clean_dataset(self, dataset_path, output_path=None):
        """Nettoie les mauvaise images du dataset (corrompues, sans visage, etc.)
        Paramètres :
        - dataset_path : chemin du dataset à nettoyer
        - output_path : chemin de sortie (si None, utilise dataset_path)
        """
        
        output_path = output_path or dataset_path
        os.makedirs(output_path, exist_ok=True)
        
        HELPERS.log(f"Nettoyage du dataset : {dataset_path}", "INFOS")
        
        deleted_count = 0
        valid_count = 0
        
        # Parcourir les dossiers de personnes
        person_folders = HELPERS.read_folder(dataset_path)[1]
        for person_folder in person_folders:
            valid = 0
            deleted = 0
            
            person_name = os.path.basename(person_folder)
            output_person_folder = os.path.join(output_path, person_name)
            os.makedirs(output_person_folder, exist_ok=True) # créer un dossier pour chaque personne dans le dossier de sortie
            
            HELPERS.log(f"🔍 Nettoyage de {person_name}...", "INFOS")
            
            image_files = HELPERS.read_folder(person_folder)[1]
            for image_file in image_files:
                
                # Validation de l'image
                if HELPERS.is_image_file(image_file) and self.validation_image(image_file):
                    HELPERS.log(f"Valid: {image_file}", "INFOS")
                    
                    # Lire l'image de manière sécurisée
                    image = HELPERS.safe_read_image(image_file)
                    if image is None:
                        HELPERS.log(f"Error reading image: {image_file}", "ERROR")
                        deleted += 1
                        continue
                    
                    # Vérifier la qualiter de l'image
                    quality = HELPERS.check_image_quality(image)
                    if not quality["is_valid"]:
                        HELPERS.log(f"⚠ {image_file} : Image de mauvaise qualité", "WARNING")
                        deleted += 1
                        continue
                    
                    # Copier l'image valide
                    output_img_path = os.path.join(output_person_folder, os.path.basename(image_file))
                    cv2.imwrite(output_img_path, image)
                    valid += 1
                else:
                    HELPERS.log(f"Invalid: {image_file}", "ERROR")
                    deleted += 1
                    continue
            HELPERS.log(f"✅ Nettoyage terminé. \n👤 {person_name} : {valid} valides ✔️, {deleted} supprimées ❌", "INFO")
            valid_count += valid
            deleted_count += deleted
        
        HELPERS.log(f"✅📂 Dataset nettoyé : {valid_count} valides ✔️, {deleted_count} supprimées ❌", "INFO")
        return {"valid": valid_count, "deleted": deleted_count}

    def validation_image(self, img_path):
        """Validation centralisée"""
        try:
            with Image.open(img_path) as img:
                img.verify()
            return cv2.imread(str(img_path)) is not None
        except Exception as e:
            self.log.append(f"Invalid: {img_path} - {e}")
            return False
    
    # AUGMENTATION
    
    def augment_dataset(self, dataset_path, target_count = 2000, output_path = None):
        """
        Augmenter chaque classe jusqu'au target_count
        Paramètres :
        - dataset_path : chemin du dataset à augmenter
        - target_count : nombre d'images souhaité par classe après augmentation
        - output_path : chemin de sortie (si None, utilise dataset_path)
        """
        
        output_path = output_path or dataset_path
        os.makedirs(output_path, exist_ok=True)
        
        HELPERS.log(f"Augmentation du dataset (target = {target_count})", "INFOS")
        
        stats = {"augmented": 0, "skipped": 0} # statistiques d'augmentation
        
        persons = HELPERS.read_folder(dataset_path)[1]
        for person in persons:
            person_name = os.path.basename(person)
            output_person_folder = os.path.join(output_path, person_name)
            os.makedirs(output_person_folder, exist_ok=True)
            
            images = HELPERS.read_folder(person)[1]
            current_count = len(images)
            
            # Vérifier si l'augmentation est nécessaire
            if current_count >= target_count:
                HELPERS.log(f"✅ {person_name} : {current_count} images (déjà suffisant)", "INFOS")
                stats["skipped"] += 1
                continue
            
            # Augmentation nécessaire
            augmentations_needed = target_count - current_count
            HELPERS.log(f"🔄 {person_name} : {current_count} images, besoin de {augmentations_needed} augmentations", "INFOS")
            
            # Copier les images existantes
            for img in images:
                output_img_path = os.path.join(output_person_folder, os.path.basename(img))
                cv2.imwrite(output_img_path, cv2.imread(img))
            
            # Générer des augmentations
            while current_count < target_count:
                img_path = random.choice(images)
                image = HELPERS.safe_read_image(img_path)
                
                if image is None:
                    HELPERS.log(f"Error reading image for augmentation: {img_path}", "ERROR")
                    continue
                
                # Générer plusieurs augmentations à partir de la même image pour accélérer le processus
                augmentation = self.augmenter_img(image)
                    
                # Enregistrer les augmentations générées
                for aug_img in augmentation:
                    if current_count >= target_count: # vérifie à chaque fois pour éviter de dépasser le target_count
                        break
                    
                    aug_img_name = f"{os.path.splitext(os.path.basename(img_path))[0]}_aug_{current_count}.jpg"
                    output_aug_path = os.path.join(output_person_folder, aug_img_name)
                    
                    cv2.imwrite(output_aug_path, aug_img)
                    current_count += 1
                    stats["augmented"] += 1
            
            HELPERS.log(f"✅ {person_name} : Augmentation terminée ({current_count} images)", "INFOS")
        
        HELPERS.log(f"✅📂 Augmentation terminée : {stats['augmented']} images générées, {stats['skipped']} classes déjà suffisantes", "INFOS")
        return stats
    
    
    # ENCODAGE
    def encode_faces(self):
        """Transformer image en vecteur"""
        
        return
    
    def save_encodings(self):
        """Sauvegarde les encodages"""
        
        return
    
    def load_encodings(self):
        """Charger les encodages"""
        
        return
    
    def train_test_split(self):
        """Divise le dataset en Train/Test/Validation"""

        return
    
    # PIPELINE GLOBAL
    
    def run_full_pipeline(self):
        """Pipeline globale"""
        
        return
    
"""======================================================================"""

class TrainTestModel():
    """
    Classe pour entraîner et évaluer des modèles de reconnaissance faciale
    Supporte SVM standard et Ensemble Learning
    """
    
    
    def __init__(self, config, model_type='ensemble'):
        """
        Initialiser le modèle
        
        Paramètres:
        - config: configuration du modèle
        - model_type: 'ensemble' ou 'svm'
        """
        from cores.EnsembleLearning import EnsembleRecognizer
        self.config = config
        self.model_type = model_type
        
        if model_type == 'ensemble':
            self.model = EnsembleRecognizer(config)
        else:
            self.model = None
        
        HELPERS.log(f"🎯 Modèle initialisé : {model_type}", "INFOS")
    
    # ====== ENTRAÎNEMENT ======
    
    def train_model(self, X_train, y_train):
        """
        Entraîne le modèle choisi
        
        Paramètres:
        - X_train: features d'entraînement
        - y_train: labels d'entraînement
        """
        if self.model_type == 'ensemble':
            return self.model.train_ensemble(X_train, y_train)
        else:
            HELPERS.log("❌ Modèle non supporté", "ERROR")
            return None
    
    def tune_model(self, X_train, y_train, cv=5):
        """
        Optimise les hyperparamètres du modèle (GridSearchCV)
        
        Paramètres:
        - X_train: features d'entraînement
        - y_train: labels d'entraînement
        - cv: nombre de folds pour la validation croisée
        """
        from sklearn.model_selection import GridSearchCV
        from sklearn.svm import SVC
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        from sklearn.pipeline import Pipeline
        
        HELPERS.log("🔍 Tuning des hyperparamètres...", "INFOS")
        
        # Encoder les labels pour SVM
        le = LabelEncoder()
        y_encoded = le.fit_transform(y_train)
        
        # Pipeline avec normalisation + SVM
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(probability=True, random_state=42))
        ])
        
        # Grille d'hyperparamètres à tester
        param_grid = {
            'svm__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],  # tester plusieurs types de noyaux
            'svm__C': [0.1, 1, 10, 100],                          # tester différentes valeurs de régularisation
            'svm__gamma': ['scale', 'auto', 0.001, 0.01]          # tester différentes stratégies de gamma pour les noyaux non linéaires
        }
        
        # Utiliser GridSearchCV pour trouver les meilleurs hyperparamètres
        grid_search = GridSearchCV(pipeline, param_grid, cv=cv, n_jobs=-1, verbose=1) # verbose=1 pour afficher la progression
        grid_search.fit(X_train, y_encoded) # entraîner le modèle sur les données d'entraînement
        
        HELPERS.log(f"✅ Meilleurs paramètres : {grid_search.best_params_}", "INFOS")
        HELPERS.log(f"   Meilleur score CV : {grid_search.best_score_:.4f}", "INFOS")
        
        return grid_search
    
    # ====== PRÉDICTION ======
    
    def predict(self, image_path):
        """
        Prédit l'identité d'une personne sur une image
        
        Paramètres:
        - image_path: chemin de l'image
        
        Retourne: dict avec 'name', 'confidence', 'is_confident'
        """
        from deepface import DeepFace
        
        HELPERS.log(f"🔍 Prédiction sur : {image_path}", "INFOS")
        
        try:
            # Extraire embeddings pour les 3 modèles
            X_ensemble = {}
            for model_name in self.config['embedding_models']:
                try:
                    embedding = DeepFace.represent(
                        img_path=image_path,
                        model_name=model_name,
                        enforce_detection=True
                    )[0]["embedding"]
                    X_ensemble[model_name] = [embedding]
                except Exception as e:
                    HELPERS.log(f"⚠️ Erreur {model_name}: {str(e)[:50]}", "WARNING")
            
            # Combiner les embeddings
            X_combined = self.model.combine_embeddings(
                {k: np.array(v) for k, v in X_ensemble.items()},
                strategy='concatenate'
            )
            
            # Prédire
            prediction = self.model.predict_single(X_combined[0], confidence_threshold=0.90)
            
            HELPERS.log(f"✅ Prédiction : {prediction['name']} ({prediction['confidence']:.2f})", "INFOS")
            
            return prediction
            
        except Exception as e:
            HELPERS.log(f"❌ Erreur prédiction : {e}", "ERROR")
            return {'name': 'Unknown', 'confidence': 0, 'is_confident': False}
    
    def evaluate_model(self, X_test, y_test):
        """
        Évalue la performance du modèle
        
        Paramètres:
        - X_test: features de test
        - y_test: labels de test
        
        Retourne: dict avec métriques
        """
        return self.model.evaluate_ensemble(X_test, y_test)
    
    def confusion_matrix(self, X_test, y_test):
        """
        Visualise la matrice de confusion
        
        Paramètres:
        - X_test: features de test
        - y_test: labels de test
        """
        return self.model.confusion_matrix_ensemble(X_test, y_test)
    
    # ====== SAUVEGARDE/CHARGEMENT ======
    
    def save_model(self, filepath='ensemble_model.pkl'):
        """
        Sauvegarde le modèle entraîné
        
        Paramètres:
        - filepath: chemin du fichier de sortie
        """
        return self.model.save_ensemble(filepath)
    
    def load_model(self, filepath='ensemble_model.pkl'):
        """
        Charge un modèle sauvegardé
        
        Paramètres:
        - filepath: chemin du fichier d'entrée
        """
        return self.model.load_ensemble(filepath)
    
    # ====== PIPELINE GLOBAL ======
    
    def run_training_pipeline(self, dataset_path, output_path='models/ensemble', 
                            test_ratio=0.15, val_ratio=0.15):
        """
        Pipeline COMPLET pour Ensemble Learning
        
        Paramètres:
        - dataset_path: chemin du dataset (organisé par personne)
        - output_path: chemin de sortie pour les modèles
        - test_ratio: proportion du test set
        - val_ratio: proportion du validation set
        
        Retourne: dict avec résultats et chemins
        """
        from cores.EnsembleLearning import EnsembleRecognizer
        from sklearn.model_selection import train_test_split as sklearn_split
        
        os.makedirs(output_path, exist_ok=True)
        
        HELPERS.log("🚀 Pipeline Ensemble Learning Complet", "INFOS")
        
        # 1. Extraire embeddings multiples
        X_ensemble, y = self.model.encode_faces_ensemble(dataset_path)
        
        # 2. Sauvegarder les embeddings
        embeddings_path = os.path.join(output_path, 'embeddings_ensemble.pkl')
        self.model.save_encodings(X_ensemble, y, embeddings_path)
        
        # 3. Combiner les embeddings
        X_combined = self.model.combine_embeddings(X_ensemble, strategy='concatenate')
        
        # 4. Split train/val/test
        X_temp, X_test, y_temp, y_test = sklearn_split(
            X_combined, y, test_size=test_ratio, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = sklearn_split(
            X_temp, y_temp, 
            test_size=val_ratio/(1-test_ratio),
            random_state=42, 
            stratify=y_temp
        )
        
        HELPERS.log(f"📊 Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}", "INFOS")
        
        # 5. Entraîner ensemble
        self.train_model(X_train, y_train)
        
        # 6. Évaluer sur validation
        HELPERS.log("📈 Évaluation Validation Set", "INFOS")
        val_metrics = self.evaluate_model(X_val, y_val)
        
        # 7. Comparer classifieurs individuels
        individual_scores = self.model.compare_classifiers(X_val, y_val)
        
        # 8. Évaluer sur test
        HELPERS.log("📈 Évaluation Test Set", "INFOS")
        test_metrics = self.evaluate_model(X_test, y_test)
        
        # 9. Matrice de confusion
        confusion_mat = self.confusion_matrix(X_test, y_test)
        
        # 10. Sauvegarder le modèle
        model_path = os.path.join(output_path, 'ensemble_model.pkl')
        self.save_model(model_path)
        
        HELPERS.log("✅ Pipeline Ensemble terminé !", "INFOS")
        
        return {
            'model_path': model_path,
            'embeddings_path': embeddings_path,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics,
            'individual_scores': individual_scores,
            'confusion_matrix': confusion_mat,
            'output_path': output_path
        }