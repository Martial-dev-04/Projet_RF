import os
import shutil
from sklearn.model_selection import train_test_split

def prepare_splits(data_dir, output_dir, val_split=0.2):
    train_list = []
    val_list = []
    all_images = []
    all_labels = []

    # Collecter toutes les images et leurs labels
    for person in os.listdir(data_dir):
        person_dir = os.path.join(data_dir, person)
        if not os.path.isdir(person_dir):
            continue
        images = [os.path.join(person, f) for f in os.listdir(person_dir)
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if len(images) < 2:
            continue
        
        for img in images:
            all_images.append(img)
            all_labels.append(person)
    
    # Split STRATIFIÉ par classe
    train_imgs, val_imgs, train_labels_split, val_labels_split = train_test_split(
        all_images, all_labels, 
        test_size=val_split, 
        random_state=42,
        stratify=all_labels  # ✅ Assure représentation égale de chaque classe
    )
    
    with open(os.path.join(output_dir, 'train.txt'), 'w') as f:
        for img, label in zip(train_imgs, train_labels_split):
            f.write(f"{os.path.join(data_dir, img)} {label}\n")
    
    with open(os.path.join(output_dir, 'val.txt'), 'w') as f:
        for img, label in zip(val_imgs, val_labels_split):
            f.write(f"{os.path.join(data_dir, img)} {label}\n")
    print(f"Fichiers train/val créés dans {output_dir}")
    print(f"Train : {len(train_imgs)} images , Val : {len(val_imgs)} images")
    
    
if __name__ == '__main__':
    # Exemple : python data_preparation.py --data_dir ../data --output_dir ../data/splits
    '''
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()  
    '''  
    
    datadir = 'C:/PROJETS/RF_dataset/DatasetOriginal_clean_aug'
    outputdir = 'C:/PROJETS/RF_dataset/DatasetOriginal_splits'
    os.makedirs(outputdir, exist_ok=True)
    prepare_splits(datadir, outputdir)