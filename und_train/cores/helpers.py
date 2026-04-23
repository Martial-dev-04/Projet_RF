"""

"""

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import face_recognition

class HELPERS():
    @staticmethod
    def log(message, level="INFO"):
        levels = {"INFO": "ℹ️", "WARNING": "⚠️", "ERROR": "❌"}
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
        
    @staticmethod
    def get_image_size(path):
        """
        Retourne la taille de l'image
        Param: path (str) - chemin de l'image
        Return: (width, height) ou (0, 0) si erreur
        """
        img = HELPERS.safe_read_image(path)
        h,w,c = img.shape
        return (w,h)
    
    @staticmethod
    def read_folder(folder_path):
        """Lire le contenu d'un dossier"""
        if not os.path.exists(folder_path):
            return (0, [])
        
        folders = []
        try:
            for el in os.listdir(folder_path):
                el_path = os.path.join(folder_path, el)
                if os.path.isdir(el_path):
                    folders.append(el_path)
                elif HELPERS.is_image_file(el_path):
                    folders.append(el_path)
        except Exception as e:
            HELPERS.log(f"Erreur read_folder: {e}", "WARNING")
        
        return (len(folders), folders)
    
    @staticmethod
    def resize_image(image, target_size=(160, 160)):
        """Redimentionne une image"""
        return cv2.resize(image, target_size)
    
    @staticmethod
    def  normalize_image(image):
        """Normalise pixels (0 => 1)"""
        return cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX)
    
    @staticmethod
    def get_brightness(image):
        """Retourne la luminosité moyenne d'une image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return np.mean(gray)
        
    @staticmethod
    def get_sharpness(image):
        """Retourne le nettété d'une image."""
        vl = cv2.Laplacian(image, cv2.CV_64F).var()
        return vl
    
    @staticmethod
    def check_image_quality(image):
        import cv2
        import numpy as np
        import face_recognition
        
        if image is None:
            return {"is_valid": False, "score": 0, "reason": "image_none"}
        img = HELPERS.safe_read_image(image)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 🔹 1. Netteté
        sharpness = HELPERS.get_sharpness(img)
        
        # 🔹 2. Luminosité
        brightness = HELPERS.get_brightness(img)
        
        # 🔹 3. Contraste
        contrast = np.std(gray)
        
        # 🔹 4. Taille
        h, w = HELPERS.get_image_size(image)
        
        # 🔹 5. Visage détecté
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = face_recognition.face_locations(rgb)
        has_face = len(faces) > 0
        
        # 🎯 Règles (à ajuster selon ton dataset)
        conditions = {
            "sharpness": sharpness > 80,
            "brightness": 50 < brightness < 200,
            "contrast": contrast > 30,
            "size": h > 100 and w > 100,
            "face_detected": has_face
        }
        
        # Score global
        score = sum(conditions.values()) / len(conditions)
        
        return {
            "is_valid": all(conditions.values()),
            "score": score,
            "details": {
                "sharpness": sharpness,
                "brightness": brightness,
                "contrast": contrast,
                "size": (h, w),
                "face_detected": has_face
            }
        }
            
            
    # EXTRACTION VISAGE
    @staticmethod
    def detect_faces(image, use_cnn=False, scale_factor=0.25):
        """
        Détecte visages avec options configurables
        - use_cnn=True pour meilleure précision (plus lent)
        - scale_factor: redimensionne pour speedup (0.25 = 4x plus rapide)
        """
        try:
            # Redimentionnemnt pour  accélérer la détection
            small_image = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor)
            
            # Détecter
            model = "cnn" if use_cnn else "hog"
            face_location = face_recognition.face_locations(small_image, model=model)
            
            # Re-scale les coordonnées des visages détectés pour correspondre à l'image originale
            face_location = [(int(top/scale_factor), int(right/scale_factor), int(bottom/scale_factor), int(left/scale_factor)) for (top, right, bottom, left) in face_location]
            
            return True #face_location
        except Exception as e:
            HELPERS.log(f"Erreur de détection: {e}", "WARNING")
            return []
        
    @staticmethod
    def extract_faces(image, face_locations=None):
        """
        Crop les visages d'une image
        Accepte face_locations pour éviter double détection
        """
        if face_locations is None:
            face_locations = HELPERS.detect_faces(image)
        
        faces = []
        face_infos = []
        for (top, right, bottom, left) in face_locations:
            # Ajouter padding pour éviter de couper les visages
            padding = 10
            top = max(0, top - padding)
            left = max(0, left - padding)
            bottom = min(image.shape[0], bottom + padding)
            right = min(image.shape[1], right + padding)
            
            face = image[top:bottom, left:right]
            
            if face.size > 0:
                faces.append(face)
                face_infos.append({"top": top, "right": right, "bottom": bottom, "left": left})
                
        HELPERS.log(f"✅ {len(faces)} visage(s) extrait(s)", "INFOS")
        return faces, face_infos
    
    
    @staticmethod    
    def align_face(image, output_size=(160,160)):
        """
        Détecte, aligne et recadre un visage avec face_recognition
        
        Params:
        - image: numpy array (BGR)
        - output_size: taille finale
        
        Return:
        - visage aligné ou None
        """
        
        import cv2
        import numpy as np
        import face_recognition
        
        if image is None:
            return None
        
        # Convertir en RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Détection visage
        face_locations = face_recognition.face_locations(rgb)
        
        if len(face_locations) == 0:
            return None
        
        # Prendre le premier visage
        top, right, bottom, left = face_locations[0]
        
        # Landmarks
        landmarks = face_recognition.face_landmarks(rgb, [(top, right, bottom, left)])
        
        if len(landmarks) == 0:
            return None
        
        landmarks = landmarks[0]
        
        # Récupérer les yeux
        left_eye = np.mean(landmarks["left_eye"], axis=0)
        right_eye = np.mean(landmarks["right_eye"], axis=0)
        
        # Calcul angle
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Centre de rotation
        center = tuple(map(int, np.mean([left_eye, right_eye], axis=0)))
        
        # Rotation
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        aligned = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
        
        # Crop après rotation
        face_crop = aligned[top:bottom, left:right]
        
        if face_crop.size == 0:
            return None
        
        # Resize final
        face_resized = cv2.resize(face_crop, output_size)
        
        return face_resized
    
    
    @staticmethod
    def show_alignment(image_path, aligned_face):
        """
        Affiche image originale vs image alignée
        """
        
        # Lire image originale
        original = cv2.imread(image_path)
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        
        # Si alignement échoue
        if aligned_face is None:
            print("❌ Alignement échoué")
            return
        
        if aligned_face.dtype != "uint8":
            aligned_face = (aligned_face * 255).astype("uint8")
            
        #aligned_rgb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
        # ⚠️ inversion BGR → RGB
        aligned_rgb = aligned_face[:, :, ::-1]
        
        # Affichage
        plt.figure(figsize=(10,5))
        
        plt.subplot(1,2,1)
        plt.imshow(original)
        plt.title("Image originale")
        plt.axis("off")
        
        plt.subplot(1,2,2)
        plt.imshow(aligned_rgb)
        plt.title("Image alignée (160x160)")
        plt.axis("off")
        
        plt.tight_layout()
        plt.show()