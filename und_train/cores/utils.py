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
import json
import matplotlib.pyplot as plt
import face_recognition
from cores.helpers import HELPERS

random.seed(42)

pat = "C:/PROJETS/Reconnaissance_faciale/Projet_RF/und_train/notebooks/quality_thresholds.json"
try:
    with open(pat, "r", encoding="utf-8") as f:
        config = json.load(f)

except PermissionError:
    raise PermissionError(
        f"Impossible d'écrire dans {pat}"
    )
    
extreme_blur_threshold = config["sharpness_threshold"] - 10
"""
config = {
    "nbr images par classe" : 2000,
    "luminosité max": 170,
    "luminosité min": 80,
    "Nettété max": 100,
    "Nettété min": 30,
    "contraste min":30
}
"""



class DatasetProcessor:
    def __init__(self, config=config):
        self.config = config
        self.log = []
    
    # ANALYSE DATASET
    
    def get_dataset_stats(self,path):
        """Retourne : nb images, résolutions, distribution classes, etc."""
        HELPERS.log(f"📊 Analyse du dataset : {os.path.basename(path)}", "INFO")
        stats = {
            "total_images": 0,
            "total_classe": 0,
            "per_class": {},
            "image_sizes": [],
            "brightness_dist": []
            }
        folders = HELPERS.read_folder(path) # récupère un tuple de (nombre_sous_dossier, liste_chemin_sous_dossier)
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
        HELPERS.log(f"🔄 Génération de {num_variants} variantes pour augmentation...", "INFO")
        transformations = []
    
        for _ in range(num_variants):
            choix = random.choice(['rotation', 'contraste', 'zoom', 'flip', 'noise'])
            
            try:
                if choix == 'rotation':
                    angle = random.uniform(-10, 10)
                    h, w = image.shape[:2]
                    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
                    result = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
                    
                elif choix == 'contraste':
                    alpha = random.uniform(1.1, 1.6)
                    beta = random.randint(-5, 5)
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
                
                if result is not None and HELPERS.detect_faces(result):
                    transformations.append(result)
                    
            except Exception as e:
                HELPERS.log(f"Erreur augmentation ({choix}): {e}", "WARNING")
                continue
        HELPERS.log(f"✅ {len(transformations)} variantes générées avec succès", "INFO")
        
        return transformations if transformations else [image]
        
    
    def view_image(self, folder_path):
        """Choisir une image au hazard dans le dossier et l'affiche"""

        folder_person = HELPERS.read_folder(folder_path)[1]
        images_exemples = []
        el = random.choice(folder_person)
        
        # Vérifie s'il s'agit d'un fichier image ou d'un dossier de personnes
        if HELPERS.is_image_file(el):
            nom = os.path.basename(folder_path)
            images_exemples.append((nom, random.choice(folder_person)))
            
            lines = 1
            columns = 1
            figsize = (20, 20)
            
        else:
            for folder in folder_person:
                fichiers = HELPERS.read_folder(folder)[1]
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
        
        # Vérifie s'il s'agit d'un fichier image ou d'un dossier de personnes
        if HELPERS.is_image_file(el):
            HELPERS.log(f"🌕 Calcule des luminosités moyennes pour {name}...", "INFO")
            fichiers = HELPERS.read_folder(file_path)[1]
            for fichier in fichiers:
                if self.validation_image(fichier):
                    image = cv2.imread(fichier)
                    lum = HELPERS.get_brightness(image)
                    brightness.append(lum)
        else:
            HELPERS.log(f"🌕 Calcule des luminosités moyennes pour {name} (Dataset)...", "INFO")
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
        
        # Vérifie s'il s'agit d'un fichier image ou d'un dossier de personnes
        if HELPERS.is_image_file(el):
            HELPERS.log(f"🌕 Calcule de la netétté des images pour {name}...", "INFO")
            fichiers = HELPERS.read_folder(file_path)[1]
            for fichier in fichiers:
                if self.validation_image(fichier):
                    image = cv2.imread(fichier)
                    lum = HELPERS.get_sharpness(image)
                    sharpness.append(lum)
        else:
            HELPERS.log(f"🌕 Calcule de la netétté des images pour {name} (Dataset)...", "INFO")
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
    
    def process_brightness(
        self,
        image,
        threshold_low=config["brightness_min"],
        threshold_high=config["brightness_max"]
    ):
        """
        Corrige la luminosité d'une image sans perdre les couleurs.

        Paramètres:
        - image : image BGR
        - threshold_low : seuil minimum de luminosité
        - threshold_high : seuil maximum de luminosité

        Retourne:
        - tuple (processed_image, status)
            status:
            - 'corrected'
            - 'kept'
        """

        HELPERS.log(
            "🔍 Vérification de la luminosité pour ajustement...","INFO")

        # Validation d'entrée
        if image is None:
            raise ValueError("Image invalide (None).")

        if len(image.shape) != 3:
            raise ValueError(
                "L'image doit être en couleur (BGR, 3 canaux)."
            )

        brightness = HELPERS.get_brightness(image)

        adjusted = image.copy()
        correction_applied = False

        # -------- CORRECTION GAMMA --------

        if brightness < threshold_low:
            HELPERS.log(
                f"⚠ Image sombre détectée "
                f"(brightness={brightness:.2f}). "
                f"Éclaircissement en cours...",
                "WARNING"
            )

            gamma = 1.5
            inv_gamma = 1.0 / gamma

            table = np.array([
                ((i / 255.0) ** inv_gamma) * 255
                for i in range(256)
            ]).astype("uint8")

            adjusted = cv2.LUT(adjusted, table)
            correction_applied = True

        elif brightness > threshold_high:
            HELPERS.log(
                f"⚠ Image trop lumineuse détectée "
                f"(brightness={brightness:.2f}). "
                f"Réduction de luminosité...",
                "WARNING"
            )

            gamma = 0.7
            inv_gamma = 1.0 / gamma

            table = np.array([
                ((i / 255.0) ** inv_gamma) * 255
                for i in range(256)
            ]).astype("uint8")

            adjusted = cv2.LUT(adjusted, table)
            correction_applied = True

        else:
            HELPERS.log(
                f"✅ Luminosité correcte "
                f"(brightness={brightness:.2f}).",
                "INFO"
            )

        # -------- CLAHE SUR LUMINANCE UNIQUEMENT --------
        # Préserve les couleurs

        lab = cv2.cvtColor(adjusted, cv2.COLOR_BGR2LAB)

        l_channel, a_channel, b_channel = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=1.3,
            tileGridSize=(8, 8)
        )

        l_channel = clahe.apply(l_channel)

        lab = cv2.merge((l_channel, a_channel, b_channel))

        adjusted = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # Déterminer le statut
        if correction_applied:
            return adjusted, "corrected"
        else: 
            adjusted = image.copy()

        return adjusted, "kept"
    
    def process_sharpness(self, image, threshold=config["sharpness_threshold"], extreme_blur_threshold=extreme_blur_threshold):
        """
        Corrige la nettété (flou) d'une image.
        
        Paramètres:
        - image: image à traiter
        - threshold: seuil de netteté pour la correction (images légèrement floues)
        - extreme_blur_threshold: seuil de flou extrême (images à ignorer)
        
        Retourne:
        - tuple (processed_image, status) où status est 'corrected', 'ignored', ou 'kept'
        """
        
        HELPERS.log(f"🔍 Vérification de la netteté (sharpness) pour ajustement...", "INFO")
        sharpness = HELPERS.get_sharpness(image)
        
        # Cas 1 : Image très floue → À IGNORER
        if sharpness < extreme_blur_threshold:
            HELPERS.log(f"❌ Image très floue détectée (sharpness={sharpness:.2f} < {extreme_blur_threshold}). Image ignorée.", "WARNING")
            return None, 'ignored'
        
        # Cas 2 : Image légèrement floue → À CORRIGER
        elif extreme_blur_threshold <= sharpness < threshold:
            HELPERS.log(f"⚠️  Image légèrement floue détectée (sharpness={sharpness:.2f}). Application d'un filtre de netteté.", "WARNING")
            
            # Appliquer un filtre de netteté
            kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]]) 
            adjusted = cv2.filter2D(image, -1, kernel)
            return adjusted, 'corrected'
        
        # Cas 3 : Image trop nette → Léger flou pour éviter les artefacts
        elif sharpness > 500:
            HELPERS.log(f"⚠️  Image très nette détectée (sharpness={sharpness:.2f}). Application d'un léger flou pour éviter les artefacts.", "WARNING")
            
            # Appliquer un léger flou pour éviter les artefacts
            adjusted = cv2.GaussianBlur(image, (3, 3), 0)
            return adjusted, 'corrected'
        
        # Cas 4 : Image avec netteté acceptable → PAS DE CORRECTION
        else:
            HELPERS.log(f"✅ Netteté acceptable (sharpness={sharpness:.2f}). Aucune correction nécessaire.", "INFO")
            return image, 'kept'
    
    def process_dataset(self, dataset_path, output_path=None, apply_brightness=True, apply_sharpness=True, skip_invalid=True):
        """
        Applique les corrections de luminosité et netteté sur TOUT le dataset.
        Traite chaque image du dataset en appliquant process_brightness() et process_sharpness().
        
        Paramètres:
        - dataset_path: chemin du dataset source (organisé par personne)
        - output_path: chemin de sortie pour les images traitées (si None, utilise dataset_path)
        - apply_brightness: booléen pour appliquer la correction de luminosité
        - apply_sharpness: booléen pour appliquer la correction de netteté
        - skip_invalid: booléen pour sauter les images invalides
        
        Retourne: dict avec statistiques du traitement
        """
        
        output_path = output_path or dataset_path
        os.makedirs(output_path, exist_ok=True)
        
        HELPERS.log(f"🔄 Traitement du dataset : {dataset_path}", "INFO")
        HELPERS.log(f"   - Correction luminosité : {'✅' if apply_brightness else '❌'}", "INFO")
        HELPERS.log(f"   - Correction netteté : {'✅' if apply_sharpness else '❌'}", "INFO")
        
        stats = {
            "total_images": 0,
            "processed": 0,
            "skipped": 0,
            "ignored": 0,
            "brightness_corrected": 0,
            "sharpness_corrected": 0,
            "errors": []
        }
        
        # Parcourir les dossiers de personnes
        person_folders = HELPERS.read_folder(dataset_path)[1]
        
        for person_folder in person_folders:
            person_name = os.path.basename(person_folder)
            output_person_folder = os.path.join(output_path, person_name)
            os.makedirs(output_person_folder, exist_ok=True)
            
            HELPERS.log(f"👤 Traitement de {person_name}...", "INFO")
            
            # Parcourir les images de la personne
            image_files = HELPERS.read_folder(person_folder)[1]
            
            for image_file in image_files:
                stats["total_images"] += 1
                
                # Validation de base
                if not HELPERS.is_image_file(image_file):
                    HELPERS.log(f"   ⏭️  {os.path.basename(image_file)} : Format non reconnu", "WARNING")
                    stats["skipped"] += 1
                    continue
                
                try:
                    # Lire l'image
                    image = HELPERS.safe_read_image(image_file)
                    if image is None:
                        raise ValueError(f"Impossible de lire : {image_file}")
                    
                    processed_image = image.copy()
                    corrections_applied = []
                    image_ignored = False
                    
                    # Appliquer la correction de netteté EN PREMIER (peut ignorer l'image)
                    if apply_sharpness:
                        try:
                            processed_image, sharpness_status = self.process_sharpness(processed_image)
                            
                            if sharpness_status == 'ignored':
                                # Image très floue → À IGNORER
                                HELPERS.log(f"   🚫 {os.path.basename(image_file)} : Image ignorée (trop floue)", "WARNING")
                                stats["ignored"] += 1
                                image_ignored = True
                                continue  # Passer à l'image suivante
                            
                            elif sharpness_status == 'corrected':
                                corrections_applied.append("netteté")
                                stats["sharpness_corrected"] += 1
                            
                        except Exception as e:
                            HELPERS.log(f"   ⚠️  Erreur netteté : {str(e)[:40]}", "WARNING")
                    
                    # Appliquer la correction de luminosité SEULEMENT si l'image n'a pas été ignorée
                    if apply_brightness and not image_ignored:
                        try:
                            processed_image, brightness_status = self.process_brightness(processed_image)

                            if brightness_status == "corrected":
                                corrections_applied.append("luminosité")
                                stats["brightness_corrected"] += 1
                        except Exception as e:
                            HELPERS.log(f"   ⚠️  Erreur luminosité : {str(e)[:40]}", "WARNING")
                    
                    # Sauvegarder l'image traitée
                    output_img_path = os.path.join(output_person_folder, os.path.basename(image_file))
                    cv2.imwrite(output_img_path, processed_image)
                    
                    # Log du traitement
                    if corrections_applied:
                        HELPERS.log(f"   ✅ {os.path.basename(image_file)} : {', '.join(corrections_applied)}", "INFO")
                    else:
                        HELPERS.log(f"   ✔️  {os.path.basename(image_file)} : Aucune correction nécessaire", "INFO")
                    
                    stats["processed"] += 1
                    
                except Exception as e:
                    error_msg = f"{os.path.basename(image_file)} : {str(e)[:50]}"
                    HELPERS.log(f"   ❌ {error_msg}", "ERROR")
                    stats["errors"].append(error_msg)
                    
                    if skip_invalid:
                        stats["skipped"] += 1
                    else:
                        # Copier l'image non traitée si on ignore pas les erreurs
                        try:
                            output_img_path = os.path.join(output_person_folder, os.path.basename(image_file))
                            cv2.imwrite(output_img_path, image)
                            stats["processed"] += 1
                        except:
                            stats["skipped"] += 1
            
            HELPERS.log(f"✅ {person_name} : Traitement terminé", "INFO")
        
        # Résumé final
        HELPERS.log(f"\n📊 Résumé du traitement :", "INFO")
        HELPERS.log(f"   Total images : {stats['total_images']}", "INFO")
        HELPERS.log(f"   Traitées : {stats['processed']} ✅", "INFO")
        HELPERS.log(f"   Ignorées (trop floues) : {stats['ignored']} 🚫", "INFO")
        HELPERS.log(f"   Skippées (erreurs/invalides) : {stats['skipped']} ⏭️", "INFO")
        HELPERS.log(f"   Corrections luminosité : {stats['brightness_corrected']}", "INFO")
        HELPERS.log(f"   Corrections netteté : {stats['sharpness_corrected']}", "INFO")
        
        if stats['errors']:
            HELPERS.log(f"   Erreurs : {len(stats['errors'])} ❌", "WARNING")
        
        HELPERS.log(f"✅ Traitement du dataset terminé !", "INFO")
        
        return stats
    
    
    # VALIDATION & NETTOYAGE
    
    def clean_dataset(self, dataset_path, output_path=None):
        """Nettoie les mauvaise images du dataset (corrompues, sans visage, etc.)
        Paramètres :
        - dataset_path : chemin du dataset à nettoyer
        - output_path : chemin de sortie (si None, utilise dataset_path)
        """
        
        output_path = output_path or dataset_path
        os.makedirs(output_path, exist_ok=True)
        
        HELPERS.log(f"Nettoyage du dataset : {dataset_path}", "INFO")
        
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
            
            HELPERS.log(f"🔍 Nettoyage de {person_name}...", "INFO")
            
            image_files = HELPERS.read_folder(person_folder)[1]
            for image_file in image_files:
                
                # Validation de l'image
                if HELPERS.is_image_file(image_file) and self.validation_image(image_file):
                    HELPERS.log(f"Valid: {os.path.basename(image_file)}", "INFO")
                    
                    # Lire l'image de manière sécurisée
                    image = HELPERS.safe_read_image(image_file)
                    
                    # Aligné le visage et redimentionné l'image à (160,160)
                    image = HELPERS.align_face(image) 
                    HELPERS.log(f"🎨 {os.path.basename(image_file)} : Visage aligné et image redimentionné à (160,160)", "INFO")
                    
                    
                    if image is None:
                        HELPERS.log(f"Error reading image: {os.path.basename(image_file)}", "ERROR")
                        deleted += 1
                        continue
                    
                    # Vérifier la qualiter de l'image
                    brightness_threshold= {'min': config["brightness_min"], "max": config["brightness_max"]}
                    
                    quality = HELPERS.check_image_quality(image, sharpness_threshold=config["sharpness_threshold"], brightness_threshold=brightness_threshold)
                    if not quality["is_valid"]:
                        HELPERS.log(f"⚠ {os.path.basename(image_file)} : Image de mauvaise qualité", "WARNING")
                        deleted += 1
                        continue
                    else:
                        HELPERS.log(f"✅ {os.path.basename(image_file)} : Qualité OK (Brightness: {quality["details"]['brightness']:.2f}, Sharpness: {quality["details"]['sharpness']:.2f})\n", "INFO")
                        
                        # Copier l'image valide
                        output_img_path = os.path.join(output_person_folder, os.path.basename(image_file))
                        cv2.imwrite(output_img_path, image)
                        valid += 1
                else:
                    HELPERS.log(f"Invalid: {os.path.basename(image_file)}", "ERROR")
                    deleted += 1
                    continue
            HELPERS.log(f"✅ Nettoyage terminé. \n👤 {person_name} : {valid} valides ✔️, {deleted} supprimées ❌\n", "INFO")
            valid_count += valid
            deleted_count += deleted
        
        HELPERS.log(f"✅📂 Dataset nettoyé : \n{valid_count} valides ✔️, {deleted_count} supprimées ❌", "INFO")
        return {"valid": valid_count, "deleted": deleted_count}

    def validation_image(self, img_path):
        """Validation centralisée"""
        try:
            with Image.open(img_path) as img:
                img.verify()
            return cv2.imread(str(img_path)) is not None
        except Exception as e:
            self.log.append(f"Invalid: {os.path.basename(img_path)} - {e}")
            return False
    
    # ====== ÉQUILIBRAGE & AUGMENTATION (BON ORDRE) ======
    
    def equilibrate_by_duplication(self, dataset_path, target_per_class=2000, output_path=None, random_seed=42):
        """
        ✅ BONNE PRATIQUE: Équilibre par DUPLICATION SIMPLE (déterministe)
        À faire AVANT la séparation train/test/validation
        
        Cette méthode duplique les images des classes minoritaires
        pour que toutes les classes aient le même nombre d'images.
        
        C'est DÉTERMINISTE (pas d'augmentation stochastique) donc
        la séparation train/test n'est PAS compromuse.
        
        Paramètres:
        - dataset_path: chemin du dataset source
        - target_per_class: nombre d'images souhaitées par classe
        - output_path: chemin de sortie (si None, utilise dataset_path)
        - random_seed: pour reproductibilité
        
        Retourne: dict avec statistiques
        """
        import shutil
        
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        output_path = output_path or dataset_path
        os.makedirs(output_path, exist_ok=True)
        
        HELPERS.log(f"⚖️  ÉQUILIBRAGE par duplication simple (target={target_per_class})", "INFO")
        HELPERS.log(f"   ✅ Déterministe = Pas de fuite de données", "INFO")
        HELPERS.log(f"   ℹ️  À faire AVANT split train/test", "INFO")
        
        stats = {
            "total_classes": 0,
            "per_class": {},
            "total_duplicated": 0,
            "duplications_needed": 0
        }
        
        # Parcourir les personnes
        person_folders = HELPERS.read_folder(dataset_path)[1]
        stats["total_classes"] = len(person_folders)
        
        for person_folder in person_folders:
            person_name = os.path.basename(person_folder)
            output_person_folder = os.path.join(output_path, person_name)
            os.makedirs(output_person_folder, exist_ok=True)
            
            # Lister les images originales
            image_files = HELPERS.read_folder(person_folder)[1]
            image_files = [f for f in image_files if HELPERS.is_image_file(f)]
            
            current_count = len(image_files)
            duplications_needed = max(0, target_per_class - current_count)
            
            HELPERS.log(f"👤 {person_name}: {current_count} → {target_per_class} (besoin {duplications_needed} duplications)", "INFO")
            
            # Copier les images originales
            for img_path in image_files:
                output_img_path = os.path.join(output_person_folder, os.path.basename(img_path))
                shutil.copy2(img_path, output_img_path)
            
            # Dupliquer les images pour atteindre le target
            if duplications_needed > 0:
                # Cycle sur les images originales pour les dupliquer
                images_to_duplicate = image_files.copy()
                random.shuffle(images_to_duplicate)
                
                duplicated = 0
                idx = 0
                while duplicated < duplications_needed:
                    img_path = images_to_duplicate[idx % len(images_to_duplicate)]
                    img_name = os.path.splitext(os.path.basename(img_path))[0]
                    img_ext = os.path.splitext(os.path.basename(img_path))[1]
                    
                    # Créer une copie avec un suffixe _dup_XXX
                    new_img_name = f"{img_name}_dup_{duplicated}{img_ext}"
                    output_img_path = os.path.join(output_person_folder, new_img_name)
                    
                    shutil.copy2(img_path, output_img_path)
                    duplicated += 1
                    idx += 1
                
                stats["total_duplicated"] += duplicated
                stats["duplications_needed"] += duplications_needed
            
            stats["per_class"][person_name] = target_per_class
        
        HELPERS.log(f"✅ Équilibrage terminé:", "INFO")
        HELPERS.log(f"   Classes: {stats['total_classes']}", "INFO")
        HELPERS.log(f"   Duplications totales: {stats['total_duplicated']}", "INFO")
        HELPERS.log(f"   Chaque classe: {target_per_class} images", "INFO")
        
        return stats
    
    
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
        
        
        
"""
=====================================================================
                Traitement de la qualité des images
=====================================================================
"""
    
import os
import cv2
import json
import numpy as np
from datetime import datetime


class ImageQualityProcessor():

    def __init__(self):
        self.metrics = {
            "sharpness": [],
            "brightness": [],
            "contrast": []
        }

    # ==========================================================
    # CALCUL DES MÉTRIQUES
    # ==========================================================

    def _compute_metrics(self, image):
        """
        Calcule :
        - netteté
        - luminosité
        - contraste
        """

        if image is None:
            raise ValueError("Image invalide.")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Netteté (Variance du Laplacien)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Luminosité moyenne
        brightness = np.mean(gray)

        # Contraste (écart-type)
        contrast = np.std(gray)

        return sharpness, brightness, contrast

    # ==========================================================
    # SUPPRESSION DES OUTLIERS
    # ==========================================================

    def _remove_outliers(self, values):
        """
        Supprime les outliers avec méthode IQR
        """

        values = np.array(values)

        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)

        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        filtered = values[
            (values >= lower_bound) &
            (values <= upper_bound)
        ]

        return filtered

    # ==========================================================
    # STATS DESCRIPTIVES
    # ==========================================================

    def _compute_summary_stats(self, values):
        """
        Retourne statistiques descriptives
        """

        values = np.array(values)

        return {
            "mean": round(float(np.mean(values)),2),
            "median": round(float(np.median(values)),2),
            "std": round(float(np.std(values)),2),
            "min": round(float(np.min(values)),2),
            "max": round(float(np.max(values)),2)
        }

    # ==========================================================
    # SAUVEGARDE CONFIG
    # ==========================================================

    def _save_thresholds(self, config, output_path):
        """
        Sauvegarde JSON
        """

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=4)

        except PermissionError:
            raise PermissionError(
                f"Impossible d'écrire dans {output_path}"
            )

    def _load_thresholds(self,output_path):
        """
        Charger JSON
        """

        try:
            with open(output_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                
            return config

        except PermissionError:
            raise PermissionError(
                f"Impossible d'écrire dans {output_path}"
            )
    # ==========================================================
    # FIT QUALITY THRESHOLDS
    # ==========================================================

    def fit_quality_thresholds(
        self,
        dataset_path,
        lower_percentile=10,
        upper_percentile=90,
        remove_outliers=True,
        save_config=True
    ):
        """
        Analyse le dataset et apprend les seuils optimaux
        """
        HELPERS.log(f"🔄Recherche des seuils optimals pour: netteté, luminosité, contraste ...", "INFO")

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(
                f"Dataset introuvable : {dataset_path}"
            )

        image_count = 0

        # Reset metrics
        self.metrics = {
            "sharpness": [],
            "brightness": [],
            "contrast": []
        }

        # ======================================================
        # PARCOURS DATASET
        # ======================================================

        for root, _, files in os.walk(dataset_path):

            for file in files:

                image_path = os.path.join(root, file)

                image = cv2.imread(image_path)

                if image is None:
                    print(f"⚠️ Image ignorée : {image_path}")
                    continue

                try:
                    sharpness, brightness, contrast = \
                        self._compute_metrics(image)

                    self.metrics["sharpness"].append(sharpness)
                    self.metrics["brightness"].append(brightness)
                    self.metrics["contrast"].append(contrast)

                    image_count += 1

                except Exception as e:
                    print(f"⚠️ Erreur métrique : {image_path} | {e}")

        # ======================================================
        # VALIDATION MINIMALE
        # ======================================================

        if image_count < 30:
            raise ValueError(
                "Dataset insuffisant (minimum 30 images recommandé)."
            )

        print(f"✅ Images analysées : {image_count}")

        # ======================================================
        # SUPPRESSION OUTLIERS
        # ======================================================

        sharpness_values = self.metrics["sharpness"]
        brightness_values = self.metrics["brightness"]
        contrast_values = self.metrics["contrast"]

        if remove_outliers:
            sharpness_values = self._remove_outliers(sharpness_values)
            brightness_values = self._remove_outliers(brightness_values)
            contrast_values = self._remove_outliers(contrast_values)

        # ======================================================
        # CALCUL DES SEUILS
        # ======================================================

        sharpness_threshold = np.percentile(
            sharpness_values,
            lower_percentile
        )

        brightness_min = np.percentile(
            brightness_values,
            5
        )

        brightness_max = np.percentile(
            brightness_values,
            95
        )

        contrast_threshold = np.percentile(
            contrast_values,
            lower_percentile
        )

        # ======================================================
        # GARDE-FOUS
        # ======================================================

        sharpness_threshold = max(15, sharpness_threshold)

        brightness_min = max(20, brightness_min)

        brightness_max = min(240, brightness_max)

        contrast_threshold = max(10, contrast_threshold)

        # ======================================================
        # STATS DESCRIPTIVES
        # ======================================================

        summary_stats = {
            "sharpness": self._compute_summary_stats(sharpness_values),
            "brightness": self._compute_summary_stats(brightness_values),
            "contrast": self._compute_summary_stats(contrast_values)
        }

        # ======================================================
        # CONFIG FINALE
        # ======================================================

        thresholds = {
            "sharpness_threshold": round(float(sharpness_threshold),2),
            "brightness_min": round(float(brightness_min),2),
            "brightness_max": round(float(brightness_max),2),
            "contrast_threshold": round(float(contrast_threshold),2),
            "dataset_size": image_count,
            "summary_stats": summary_stats,
            "computed_at": datetime.now().isoformat()
        }

        # ======================================================
        # SAUVEGARDE
        # ======================================================

        if save_config:
            self._save_thresholds(
                thresholds,
                "quality_thresholds.json"
            )

            print("✅ Seuils sauvegardés : quality_thresholds.json")

        # ======================================================
        # LOG FINAL
        # ======================================================

        print("\n📊 Seuils appris :")
        print(f"Netteté min : {sharpness_threshold:.2f}")
        print(f"Luminosité min : {brightness_min:.2f}")
        print(f"Luminosité max : {brightness_max:.2f}")
        print(f"Contraste min : {contrast_threshold:.2f}")

        return thresholds
    