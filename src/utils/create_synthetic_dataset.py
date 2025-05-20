"""
Script d'urgence pour créer un mini-dataset synthétique complet.
Ce script génère des images synthétiques, des descriptions, et extrait les caractéristiques
pour créer un ensemble de données minimal garanti fonctionnel.
"""
import os
import sys
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.text import Tokenizer
from PIL import Image

def create_directory(directory):
    """Crée un répertoire s'il n'existe pas."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Répertoire créé: {directory}")
    return directory

def create_synthetic_image(image_path, color):
    """Crée une image synthétique simple (carré coloré)."""
    # Créer une image de 299x299 pixels (taille requise par InceptionV3)
    img_array = np.zeros((299, 299, 3), dtype=np.uint8)
    img_array[:, :] = color
    
    # Sauvegarder l'image
    Image.fromarray(img_array).save(image_path)
    print(f"Image créée: {image_path}")

def extract_features(image_path, model):
    """Extrait les caractéristiques d'une image avec InceptionV3."""
    # Charger et prétraiter l'image
    img = Image.open(image_path)
    img = img.resize((299, 299))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    # Extraire les caractéristiques
    feature = model.predict(img_array, verbose=0)
    
    return feature[0]

def create_minimal_dataset(base_dir, num_images=10):
    """
    Crée un ensemble de données minimal pour l'entraînement.
    
    Args:
        base_dir (str): Répertoire de base pour les données
        num_images (int): Nombre d'images à créer
        
    Returns:
        tuple: Chemins vers les répertoires et fichiers créés
    """
    print(f"\n=== CRÉATION D'UN MINI-DATASET SYNTHÉTIQUE ({num_images} IMAGES) ===")
    
    # Créer les répertoires
    data_dir = create_directory(os.path.join(base_dir, "data"))
    minimal_dir = create_directory(os.path.join(data_dir, "minimal"))
    images_dir = create_directory(os.path.join(minimal_dir, "images"))
    text_dir = create_directory(os.path.join(minimal_dir, "text"))
    features_dir = create_directory(os.path.join(minimal_dir, "features"))
    model_dir = create_directory(os.path.join(data_dir, "models"))
    
    # Définir les couleurs pour les images synthétiques
    colors = [
        [255, 0, 0],    # Rouge
        [0, 255, 0],    # Vert
        [0, 0, 255],    # Bleu
        [255, 255, 0],  # Jaune
        [255, 0, 255],  # Magenta
        [0, 255, 255],  # Cyan
        [128, 0, 0],    # Marron
        [0, 128, 0],    # Vert foncé
        [0, 0, 128],    # Bleu foncé
        [128, 128, 0],  # Olive
    ]
    
    # Créer les descriptions pour chaque couleur
    color_names = ["rouge", "vert", "bleu", "jaune", "magenta", "cyan", "marron", "vert foncé", "bleu foncé", "olive"]
    
    # Créer les images synthétiques
    image_ids = []
    for i in range(num_images):
        color_idx = i % len(colors)
        image_id = f"synthetic_{i+1:04d}"
        image_ids.append(image_id)
        
        # Créer l'image
        image_path = os.path.join(images_dir, f"{image_id}.jpg")
        create_synthetic_image(image_path, colors[color_idx])
    
    # Créer le fichier de descriptions (Flickr8k.token.txt)
    descriptions_path = os.path.join(text_dir, "Flickr8k.token.txt")
    with open(descriptions_path, "w") as f:
        for i, image_id in enumerate(image_ids):
            color_idx = i % len(colors)
            color_name = color_names[color_idx]
            
            # Écrire 3 descriptions différentes pour chaque image
            f.write(f"{image_id}.jpg un carré de couleur {color_name} sur fond noir\n")
            f.write(f"{image_id}.jpg une forme {color_name} au centre de l'image\n")
            f.write(f"{image_id}.jpg un objet {color_name} simple sur un fond uni\n")
    
    print(f"Fichier de descriptions créé: {descriptions_path}")
    
    # Créer les fichiers de split (train, val, test)
    train_path = os.path.join(text_dir, "Flickr_8k.trainImages.txt")
    val_path = os.path.join(text_dir, "Flickr_8k.devImages.txt")
    test_path = os.path.join(text_dir, "Flickr_8k.testImages.txt")
    
    # Utiliser toutes les images pour l'entraînement, la validation et le test
    with open(train_path, "w") as f:
        for image_id in image_ids:
            f.write(f"{image_id}.jpg\n")
    
    # Copier le même contenu pour val et test
    import shutil
    shutil.copy2(train_path, val_path)
    shutil.copy2(train_path, test_path)
    
    print(f"Fichiers de split créés: {train_path}, {val_path}, {test_path}")
    
    # Charger le modèle InceptionV3 pour l'extraction de caractéristiques
    print("Chargement du modèle InceptionV3...")
    model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
    
    # Extraire les caractéristiques des images
    print("Extraction des caractéristiques...")
    features = {}
    for image_id in image_ids:
        image_path = os.path.join(images_dir, f"{image_id}.jpg")
        features[image_id] = extract_features(image_path, model)
    
    # Sauvegarder les caractéristiques
    train_features_path = os.path.join(features_dir, "train_features.npy")
    val_features_path = os.path.join(features_dir, "val_features.npy")
    test_features_path = os.path.join(features_dir, "test_features.npy")
    
    np.save(train_features_path, features)
    np.save(val_features_path, features)
    np.save(test_features_path, features)
    
    print(f"Caractéristiques sauvegardées: {train_features_path}, {val_features_path}, {test_features_path}")
    
    # Créer le tokenizer
    descriptions = {}
    for i, image_id in enumerate(image_ids):
        color_idx = i % len(colors)
        color_name = color_names[color_idx]
        
        descriptions[image_id] = [
            f"startseq un carré de couleur {color_name} sur fond noir endseq",
            f"startseq une forme {color_name} au centre de l'image endseq",
            f"startseq un objet {color_name} simple sur un fond uni endseq"
        ]
    
    # Collecter toutes les descriptions
    all_desc = []
    for image_id, desc_list in descriptions.items():
        for desc in desc_list:
            all_desc.append(desc)
    
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_desc)
    
    # Sauvegarder le tokenizer
    tokenizer_path = os.path.join(model_dir, "tokenizer.pkl")
    with open(tokenizer_path, "wb") as f:
        pickle.dump(tokenizer, f)
    
    print(f"Tokenizer créé et sauvegardé: {tokenizer_path}")
    print(f"Taille du vocabulaire: {len(tokenizer.word_index) + 1}")
    
    print(f"\n=== MINI-DATASET SYNTHÉTIQUE CRÉÉ AVEC SUCCÈS ===")
    print(f"Images: {len(image_ids)}")
    print(f"Répertoire d'images: {images_dir}")
    print(f"Répertoire de textes: {text_dir}")
    print(f"Répertoire de caractéristiques: {features_dir}")
    
    return minimal_dir, images_dir, text_dir, features_dir

def main():
    """
    Fonction principale pour créer un mini-dataset synthétique.
    """
    print("=== CRÉATION D'UN MINI-DATASET SYNTHÉTIQUE ===")
    
    # Définir le répertoire de base (répertoire courant)
    base_dir = os.getcwd()
    print(f"Répertoire de base: {base_dir}")
    
    # Créer le mini-dataset synthétique
    minimal_dir, images_dir, text_dir, features_dir = create_minimal_dataset(base_dir, num_images=10)
    
    if minimal_dir:
        print("\n=== INSTRUCTIONS POUR L'ENTRAÎNEMENT ===")
        print("Pour entraîner le modèle avec ce mini-dataset, exécutez:")
        print("python src/training/train_simplified.py")
        print("\nLe script utilisera automatiquement le mini-dataset créé.")
    else:
        print("\nÉCHEC: Impossible de créer le mini-dataset synthétique.")

if __name__ == "__main__":
    main()
