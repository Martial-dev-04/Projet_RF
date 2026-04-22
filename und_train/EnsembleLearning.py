"""
Workaround pour éviter circular imports
"""
import sys
import importlib

# Vider le cache des modules si nécessaire
if 'EnsembleLearning' in sys.modules:
    del sys.modules['EnsembleLearning']
"""======================================================================"""

from helpers import HELPERS
import os
import numpy as np
from matplotlib import pyplot as plt


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
    
    # ====== EXTRACTION D'EMBEDDINGS MULTIPLES ======
    
    def encode_faces_ensemble(self, dataset_path, enforce_detection=True):
        """
        Extrait les embeddings de TOUS les modèles pour chaque image
        
        Paramètres:
        - dataset_path: chemin du dataset organisé par personne
        - enforce_detection: strict ou permissif pour la détection de visage
        
        Retourne: (X_ensemble, y)
        - X_ensemble = dict {model_name: embeddings_array}
        - y = labels array
        """
        from deepface import DeepFace
        
        HELPERS.log(f"🔄 Extraction des embeddings pour {len(self.config['embedding_models'])} modèles...", "INFOS")
        
        X_ensemble = {model: [] for model in self.config['embedding_models']}
        y_labels = []
        errors_per_model = {model: 0 for model in self.config['embedding_models']}
        processed_count = 0
        
        # Parcourir les personnes
        for person in os.listdir(dataset_path):
            person_path = os.path.join(dataset_path, person)
            
            if not os.path.isdir(person_path):
                continue
            
            HELPERS.log(f"👤 Traitement : {person}...", "INFOS")
            
            # Parcourir les images de la personne
            for img_file in os.listdir(person_path):
                if not HELPERS.is_image_file(img_file):
                    continue
                
                img_path = os.path.join(person_path, img_file)
                embeddings_extracted = 0
                
                # Extraire avec CHAQUE modèle
                for model_name in self.config['embedding_models']:
                    try:
                        embedding = DeepFace.represent(
                            img_path=img_path,
                            model_name=model_name,
                            enforce_detection=enforce_detection
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
            HELPERS.log(f"✅ {model_name}: {len(embeddings)} embeddings extraits", "INFOS")
        
        HELPERS.log(f"✅📂 Total: {processed_count} images traitées", "INFOS")
        
        return X_ensemble_np, np.array(y_labels)
    
    def combine_embeddings(self, X_ensemble, strategy='concatenate'):
        """
        Combine les embeddings de plusieurs modèles
        
        Paramètres:
        - X_ensemble: dict {model_name: embeddings}
        - strategy: 'concatenate', 'average', ou 'pca'
        
        Retourne: X_combined (array)
        """
        HELPERS.log(f"🔗 Combination des embeddings (stratégie: {strategy})", "INFOS")
        
        if strategy == 'concatenate':
            # Normaliser chaque modèle puis concaténer
            X_combined = []
            for model_name in self.config['embedding_models']:
                embeddings = X_ensemble[model_name]
                # Normaliser L2
                embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
                X_combined.append(embeddings_norm)
            
            X_combined = np.concatenate(X_combined, axis=1)
            HELPERS.log(f"✅ Embeddings combinés : shape {X_combined.shape}", "INFOS")
        
        elif strategy == 'average':
            # Moyenne des embeddings normalisés
            embeddings_list = []
            for model_name in self.config['embedding_models']:
                embeddings = X_ensemble[model_name]
                embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
                embeddings_list.append(embeddings_norm)
            
            X_combined = np.mean(embeddings_list, axis=0)
            HELPERS.log(f"✅ Embeddings moyennés : shape {X_combined.shape}", "INFOS")
        
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
            HELPERS.log(f"✅ PCA appliqué : {X_concat.shape} → {X_combined.shape}", "INFOS")
        
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
        HELPERS.log(f"✅ Encodages ensemble sauvegardés : {filepath}", "INFOS")
        return filepath
    
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
        
        HELPERS.log(f"✅ Encodages ensemble chargés : {len(y)} labels", "INFOS")
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
        
        HELPERS.log(f"🚀 Entraînement de {len(self.config['classifiers'])} classifieurs...", "INFOS")
        
        # Encoder les labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y_train)
        
        # Normaliser les features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_train)
        
        # Entraîner chaque classifieur
        if 'svm' in self.config['classifiers']:
            HELPERS.log("  → Entraînement SVM (RBF)...", "INFOS")
            self.classifiers['svm'] = SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=42
            )
            self.classifiers['svm'].fit(X_scaled, y_encoded)
        
        if 'knn' in self.config['classifiers']:
            HELPERS.log("  → Entraînement KNN (k=5)...", "INFOS")
            self.classifiers['knn'] = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
            self.classifiers['knn'].fit(X_scaled, y_encoded)
        
        if 'rf' in self.config['classifiers']:
            HELPERS.log("  → Entraînement Random Forest (100 trees)...", "INFOS")
            self.classifiers['rf'] = RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                random_state=42,
                n_jobs=-1
            )
            self.classifiers['rf'].fit(X_scaled, y_encoded)
        
        HELPERS.log(f"✅ {len(self.classifiers)} classifieurs entraînés", "INFOS")
        
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
        
        HELPERS.log("📊 Évaluation de l'Ensemble...", "INFOS")
        
        y_encoded = self.label_encoder.transform(y_test)
        predictions, confidences = self.predict_ensemble(X_test)
        
        # Métriques
        accuracy = accuracy_score(y_encoded, predictions)
        precision = precision_score(y_encoded, predictions, average='weighted', zero_division=0)
        recall = recall_score(y_encoded, predictions, average='weighted', zero_division=0)
        f1 = f1_score(y_encoded, predictions, average='weighted', zero_division=0)
        
        HELPERS.log(f"✅ Accuracy:  {accuracy:.4f}", "INFOS")
        HELPERS.log(f"   Precision: {precision:.4f}", "INFOS")
        HELPERS.log(f"   Recall:    {recall:.4f}", "INFOS")
        HELPERS.log(f"   F1-Score:  {f1:.4f}", "INFOS")
        
        mean_confidence = np.mean(confidences)
        HELPERS.log(f"   Confiance moyenne: {mean_confidence:.4f}", "INFOS")
        
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
        
        HELPERS.log("🔍 Comparaison des classifieurs individuels...", "INFOS")
        
        y_encoded = self.label_encoder.transform(y_test)
        X_scaled = self.scaler.transform(X_test)
        
        results = {}
        for clf_name, clf in self.classifiers.items():
            preds = clf.predict(X_scaled)
            acc = accuracy_score(y_encoded, preds)
            results[clf_name] = acc
            HELPERS.log(f"  {clf_name.upper()}: {acc:.4f}", "INFOS")
        
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
        
        HELPERS.log("📉 Génération matrice de confusion...", "INFOS")
        
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
    
    def save_ensemble(self, filepath='ensemble_model.pkl'):
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
        HELPERS.log(f"✅ Modèle ensemble sauvegardé : {filepath}", "INFOS")
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
        
        HELPERS.log(f"✅ Modèle ensemble chargé : {filepath}", "INFOS")
        return True