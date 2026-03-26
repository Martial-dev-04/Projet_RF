from PIL import Image
import cv2
import os
import numpy as np
import random
import matplotlib.pyplot as plt

random.seed(42)

config = {}

class HELPERS():
    def log(message, level):
        
        return
    
    def validate_path(path):
        """Vérifie si le dossier/fichier existe"""
        
        return
    
    def is_image_file(file):
        """Vérifie l'extension du fichier"""
        
        return
    
    def safe_read_image(path):
        """lit image sans crash"""
        
        return

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
            fichiers = self.read_folder(path_folder)    # récupère un tuple de (nombre_fichiers, liste_chemin_fichier)
            
            stats["total_images"] += fichiers[0]        # (incrémente) ajoute nombre_fichiers 
            
            per_class = stats["per_class"]
            per_class[os.path.basename(path_folder)] = fichiers[0]  # ajoute le paire clé : valeur (Nom_dossier : nombre_fichier) au dictionnaire "per_classe"
            
        return stats
    
    
    def read_folder(self, folder_path):
        """Lire le contenu d'un dossier"""
        if not os.path.exists(folder_path):
            print(f"❌Le dossier 📁 {folder_path} n'existe pas.")
            return []
        else :
            folders = [os.path.join(folder_path, el) for el in os.listdir(folder_path) 
            if  os.path.isdir(os.path.join(folder_path, el)) or self.validation_image(os.path.join(folder_path, el)) and el.lower().endswith(('.jpg','.png'))
            ]
            return (len(folders), folders)
    
    def get_brightness(self, image):
        """Retourne la luminosité moyenne d'une image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return np.mean(gray)
        

    def get_sharpness(self, image):
        """Retourne le nettété d'une image."""
        vl = cv2.Laplacian(image, cv2.CV_64F).var()
        return vl
    
    def augmenter_img(self):
        """Augemente le nombre d'images au seuil maximale."""

        return
    
    def view_image(self, folder_path):
        """Choisir une image au hazard dans le dossier et l'affiche"""

        folder_person = self.read_folder(folder_path)[1]
        images_exemples = []
        el = random.choice(folder_person)
        
        if el.lower().endswith(('.jpg', '.png')):
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
        el = random.choice(self.read_folder(file_path)[1])
        
        if el.lower().endswith(('.jpg', '.png')):
            print(f"🌕 Calcule des luminosités moyennes pour {name}...")
            fichiers = self.read_folder(file_path)[1]
            for fichier in fichiers:
                if self.validation_image(fichier):
                    image = cv2.imread(fichier)
                    lum = self.get_brightness(image)
                    brightness.append(lum)
        else:
            print(f"🌕 Calcule des luminosités moyennes pour {name} (Dataset)...")
            name  = os.path.basename(file_path)
            folders = self.read_folder(file_path)[1]
            for folder in folders:
                fichiers = self.read_folder(folder)[1]
                for fichier in fichiers:
                    if self.validation_image(fichier):
                        image = cv2.imread(fichier)
                        lum = self.get_brightness(image)
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
        el = random.choice(self.read_folder(file_path)[1])
        
        if el.lower().endswith(('.jpg', '.png')):
            print(f"🌕 Calcule de la netétté des images pour {name}...")
            fichiers = self.read_folder(file_path)[1]
            for fichier in fichiers:
                if self.validation_image(fichier):
                    image = cv2.imread(fichier)
                    lum = self.get_sharpness(image)
                    sharpness.append(lum)
        else:
            print(f"🌕 Calcule de la netétté des images pour {name} (Dataset)...")
            name  = os.path.basename(file_path)
            folders = self.read_folder(file_path)[1]
            for folder in folders:
                fichiers = self.read_folder(folder)[1]
                for fichier in fichiers:
                    if self.validation_image(fichier):
                        image = cv2.imread(fichier)
                        lum = self.get_sharpness(image)
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
    
    def get_image_size(self):
        """Retourne la taille de l'image"""

        return
    
    # PRÉTRAITEMENT
    
    def process_brightness(image, threshold_low, threshold_high):
        """Corrige la luminosité des images."""

        return
    
    def process_sharpness(image, threshold_low, threshold_high):
        """Corrige la nettété (flou) d'une image"""

        return    
    def resize_image(self):
        """Redimentionne une image"""

        return
    
    def  normalize_image(self):
        """Normalise pixels (0 => 1)"""
        
        return
    
    # VALIDATION & NETTOYAGE
    
    def clean_dataset(self):
        """Nettoie les mauvaise images du dataset"""
        
        return

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
    
    def augment_dataset(self):
        """Ramener chaque classe du dataset au seuil max"""
        
        return
    
    # EXTRACTION VISAGE
    
    def detect_faces(self):
        """Detecte de visage sur une image"""
        
        return
    
    def extract_faces(self):
        """Crop le visage uniquement"""
        
        return
    
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
    def __init__(self, config, model):
        self.config = config
        self.model = model
        
    # ENTRAÎNEMENT
    
    def train_model(self):
        """Entraine le model choisir"""
        
        return
    
    def tune_model(self):
        """Optimise les hyperparamètres du modèle"""
        
        return
    
    # PRÉDICTION
    
    def predict(self, image):
        """Reconnaitre une personne"""
        
        return
    
    def evaluate_model(self):
        """Evalue la performance du modèle"""
        
        return
    
    def confusion_matrix(self):
        """Visualiser les erreurs de calssifications"""
        
        return
    
    # SAUVEGARDE
    
    def save_model(self):
        """Sauvegarde le modèle"""
        
        return
    
    def load_model(self):
        """Charger le modèle"""
        
        return
    
    # PIPELINE GLOBAL
    
    def run_training_pipeline(self):
        """Pipeline globale"""
        
        return