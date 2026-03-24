- **Découpage des scripts des fichiers `train_test.ipynb` et `und_dataset.ipynb` en de petites fonctions pour faciliter la lisibilité du code**
    - Fichier `und_datset.ipynb` :

        - Créer :
          - fonction `lire_dataset(dataset_path)`: parcourir le dataset et renvoir une liste des dossiers que contient ce dernier et affiche le nombre total de dossier.

          - fonction `lire_dossier(liste_dossier)` : parcourir chaque dossier, affiche le nombres de fichier que contient chaque dossier, le nombre total de fichier pour tout les dossiers parcourir, affiche une image choisir au hazard dans chaque dossier, puis renvoir un dictionnaire qui a pour clé (`nom_du_dossier`) et pour valeur (`liste_fichiers`)

          - fonction `lire_fichier(chemin_fichier)` : vérifie l'extension du fichier et si le fichier est lisible ou pas et renvoir le fichier. 