import os
import cv2

class HELPERS():
    @staticmethod
    def log(message, level="INFO"):
        levels = {"INFOS": "ℹ️", "WARNING": "⚠️", "ERROR": "❌"}
        print(f"{levels.get(level, '')} [{level}] {message}" )
    
    @staticmethod
    def validate_path(path):
        """Vérifie si le dossier/fichier existe"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Chemin introuvable : {path}")
        return True
    
    @staticmethod
    def is_image_file(file):
        """Vérifie l'extension du fichier"""
        return file.lower().endswith((".jpg", ".jpeg", ".png"))
    
    @staticmethod
    def safe_read_image(path):
        """lit image sans crash"""
        try:
            img = cv2.imread(path)
            if img is None :
                raise ValueError(f"Impossible de lire : {path}")
            return img
        except Exception as e :
            HELPERS.log(str(e), "ERROR")
            return None