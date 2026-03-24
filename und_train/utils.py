from PIL import Image
import cv2
import os
import numpy as np
import random
import matplotlib.pyplot as plt

random.seed(42)

class DatasetProcessor:
    def __init__(self, config=0):
        self.config = config
        self.log = []
    
    def get_dataset_stats(path):
        """Retourne : nb images, résolutions, distribution classes, etc."""
        stats = {
            "total_images": 0,
            "per_class": {},
            "image_sizes": [],
            "brightness_dist": []
            }
        

        return stats
    def validation_image(self, img_path):
        """Validation centralisée"""
        try:
            with Image.open(img_path) as img:
                img.verify()
            return cv2.imread(str(img_path)) is not None
        except Exception as e:
            self.log.append(f"Invalid: {img_path} - {e}")
            return False
    
    def read_folder(self, folder_path):
        """Lire le contenu d'un dossier"""
        if not os.path.exists(folder_path):
            print(f"❌Le dossier 📁 {folder_path} n'existe pas.")
            return []
        else :
            return [os.path.join(folder_path, el) for el in os.listdir(folder_path) 
            if self.validation_image(os.path.join(folder_path, el)) and os.path.isdir(os.path.join(folder_path, el)) or el.lower().endswith(('.jpg','.png'))
            ]
    
    def get_brightness(self, image):
        """Retourne la luminosité moyenne d'une image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return np.mean(gray)
        

    def get_sharpness(self, image):
        """Retourne le nettété d'une image."""

        return
    
    def augmenter_img(self):
        """Augemente le nombre d'images au seuil maximale."""

        return
    
    def view_image(self, folder_path):
        """Choisir une image au hazard dans le dossier et l'affiche"""
        

        folder_person = self.read_folder(folder_path)
        nom = os.path.basename(folder_path)
        
        
        images_exemples = [(nom, random.choice(folder_person))]

        # Afficher les images d'exemple
        if images_exemples:
            fig, axes = plt.subplots(1, len(images_exemples), figsize=(50, 20))

            axes = np.array(axes).reshape(-1)  # force en tableau 1D

            for i, (nom, path) in enumerate(images_exemples):
                img = Image.open(path)
                axes[i].imshow(img)
                axes[i].set_title(nom, fontsize=10)
                axes[i].axis('off')
            plt.tight_layout()
            plt.show()
        
        return folder_person
        
    
    def brightness_distribution(self):
        """Affiche la distribution de la luminosité"""

        return
    
    def sharpness_distribution(self):
        """Affiche la distribution de la nettété"""

        return
    
    def process_brightness(image, threshold_low, threshold_high):
        """Corrige la luminosité des images."""

        return
    
    def process_sharpness(image, threshold_low, threshold_high):
        """Corrige la nettété d'une image"""

        return
    
    def get_image_size(self):
        """Retourne la taille de l'image"""

        return
    
    def resize_img(self):
        """Redimentionne une image"""

        return
    
    def train_test_split(self):
        """Divise le dataset en Train/Test/Validation"""

        return