"""
Script tout-en-un pour créer un ensemble de données cohérent pour l'entraînement.
Ce script télécharge un petit sous-ensemble d'images Flickr8k, extrait les caractéristiques,
et prépare un ensemble de données minimal garanti fonctionnel.
"""
import os
import sys
import requests
import zipfile
import shutil
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from collections import defaultdict

def create_directory(directory):
    """Crée un répertoire s'il n'existe pas."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Répertoire créé: {directory}")
    return directory

def download_file(url, destination):
    """Télécharge un fichier depuis une URL."""
    if os.path.exists(destination):
        print(f"Fichier déjà téléchargé: {destination}")
        return destination
    
    print(f"Téléchargement de {url}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192
    
    with open(destination, 'wb') as file:
        for data in response.iter_content(block_size):
            file.write(data)
    
    print(f"Téléchargement terminé: {destination}")
    return destination

def extract_zip(zip_path, extract_to):
    """Extrait un fichier zip."""
    if not os.path.exists(extract_to):
        print(f"Extraction de {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Extraction terminée: {extract_to}")
    else:
        print(f"Répertoire d'extraction existe déjà: {extract_to}")
    return extract_to

def load_descriptions(filename):
    """
    Charge et parse les descriptions des images.
    
    Args:
        filename (str): Chemin vers le fichier de descriptions
        
    Returns:
        dict: Dictionnaire des descriptions par image_id
    """
    if not os.path.exists(filename):
        print(f"Fichier de descriptions non trouvé: {filename}")
        return {}
        
    # Ouvrir le fichier
    file = open(filename, 'r')
    # Lire toutes les lignes
    doc = file.read()
    file.close()
    
    # Traiter chaque ligne
    mapping = defaultdict(list)
    for line in doc.split('\n'):
        # Ignorer les lignes vides
        if len(line) < 2:
            continue
        # Diviser l'identifiant de l'image et la description
        tokens = line.split()
        if len(tokens) < 2:
            continue
            
        image_id, image_desc = tokens[0], ' '.join(tokens[1:])
        # Extraire l'identifiant de l'image sans l'extension
        image_id = image_id.split('.')[0]
        # Convertir la description en minuscules
        image_desc = image_desc.lower()
        # Ajouter un marqueur de début et de fin
        image_desc = 'startseq ' + image_desc + ' endseq'
        # Stocker la description
        mapping[image_id].append(image_desc)
    
    print(f"Nombre de descriptions chargées: {len(mapping)}")
    return mapping

def extract_features(image_paths):
    """
    Extrait les caractéristiques des images à l'aide d'InceptionV3.
    
    Args:
        image_paths (dict): Dictionnaire des chemins d'images par image_id
        
    Returns:
        dict: Caractéristiques extraites par image_id
    """
    # Charger le modèle InceptionV3 pré-entraîné sans la couche de classification
    model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
    
    # Dictionnaire pour stocker les caractéristiques
    features = {}
    
    # Traiter les images une par une
    for i, (image_id, image_path) in enumerate(image_paths.items()):
        try:
            # Charger et prétraiter l'image
            img = load_img(image_path, target_size=(299, 299))
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = preprocess_input(img_array)
            
            # Extraire les caractéristiques
            feature = model.predict(img_array, verbose=0)
            
            # Stocker les caractéristiques
            features[image_id] = feature[0]
            
            # Afficher la progression
            print(f"Traitement de l'image {i+1}/{len(image_paths)}: {image_id}")
        except Exception as e:
            print(f"Erreur lors du traitement de l'image {image_id}: {str(e)}")
    
    print(f"Caractéristiques extraites pour {len(features)}/{len(image_paths)} images.")
    return features

def create_minimal_dataset(base_dir, num_images=20):
    """
    Crée un ensemble de données minimal pour l'entraînement.
    
    Args:
        base_dir (str): Répertoire de base pour les données
        num_images (int): Nombre d'images à inclure
        
    Returns:
        tuple: Chemins vers les répertoires et fichiers créés
    """
    print(f"\n=== CRÉATION D'UN ENSEMBLE DE DONNÉES MINIMAL ({num_images} IMAGES) ===")
    
    # Créer les répertoires
    data_dir = create_directory(os.path.join(base_dir, "data"))
    minimal_dir = create_directory(os.path.join(data_dir, "minimal"))
    images_dir = create_directory(os.path.join(minimal_dir, "images"))
    text_dir = create_directory(os.path.join(minimal_dir, "text"))
    features_dir = create_directory(os.path.join(minimal_dir, "features"))
    model_dir = create_directory(os.path.join(data_dir, "models"))
    
    # Télécharger et extraire les fichiers texte de Flickr8k
    text_url = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip"
    text_zip = download_file(text_url, os.path.join(data_dir, "Flickr8k_text.zip"))
    flickr_text_dir = extract_zip(text_zip, os.path.join(data_dir, "flickr8k_text"))
    
    # Charger les descriptions
    descriptions_path = os.path.join(flickr_text_dir, "Flickr8k.token.txt")
    descriptions = load_descriptions(descriptions_path)
    
    if not descriptions:
        # Essayer un autre chemin possible
        descriptions_path = os.path.join(flickr_text_dir, "Flickr8k_text", "Flickr8k.token.txt")
        descriptions = load_descriptions(descriptions_path)
    
    if not descriptions:
        print("ERREUR: Impossible de charger les descriptions.")
        return None
    
    # Sélectionner un sous-ensemble d'images
    selected_ids = list(descriptions.keys())[:num_images]
    print(f"Images sélectionnées: {selected_ids}")
    
    # Télécharger les images sélectionnées individuellement
    image_paths = {}
    
    for image_id in selected_ids:
        # Télécharger l'image depuis Flickr
        try:
            # Utiliser l'API Flickr pour obtenir l'URL de l'image
            flickr_url = f"https://live.staticflickr.com/{image_id.split('_')[0]}/{image_id}.jpg"
            image_path = os.path.join(images_dir, f"{image_id}.jpg")
            
            # Télécharger l'image
            if not os.path.exists(image_path):
                print(f"Téléchargement de l'image {image_id}...")
                response = requests.get(flickr_url, stream=True)
                if response.status_code == 200:
                    with open(image_path, 'wb') as file:
                        for chunk in response.iter_content(1024):
                            file.write(chunk)
                    print(f"Image téléchargée: {image_path}")
                    image_paths[image_id] = image_path
                else:
                    print(f"Erreur lors du téléchargement de l'image {image_id}: {response.status_code}")
            else:
                print(f"Image déjà téléchargée: {image_path}")
                image_paths[image_id] = image_path
        except Exception as e:
            print(f"Erreur lors du téléchargement de l'image {image_id}: {str(e)}")
    
    # Si aucune image n'a pu être téléchargée, utiliser des images de test
    if not image_paths:
        print("AVERTISSEMENT: Aucune image n'a pu être téléchargée depuis Flickr.")
        print("Création d'images de test...")
        
        # Créer des images de test
        for i in range(num_images):
            image_id = f"test_{i:04d}"
            image_path = os.path.join(images_dir, f"{image_id}.jpg")
            
            # Créer une image de test (un carré coloré)
            img_array = np.zeros((299, 299, 3), dtype=np.uint8)
            color = np.random.randint(0, 256, 3)
            img_array[:, :] = color
            
            # Sauvegarder l'image
            from PIL import Image
            Image.fromarray(img_array).save(image_path)
            
            # Ajouter l'image au dictionnaire
            image_paths[image_id] = image_path
            
            # Créer des descriptions pour cette image
            descriptions[image_id] = [
                f"startseq a {['red', 'green', 'blue', 'yellow'][i % 4]} square in the image endseq",
                f"startseq this is a {['small', 'large', 'colorful', 'bright'][i % 4]} square endseq"
            ]
        
        selected_ids = list(image_paths.keys())
    
    # Extraire les caractéristiques des images
    features = extract_features(image_paths)
    
    # Vérifier que nous avons des caractéristiques
    if not features:
        print("ERREUR: Aucune caractéristique extraite.")
        return None
    
    # Créer un fichier de descriptions minimal
    mini_desc_path = os.path.join(text_dir, 'Flickr8k.token.txt')
    with open(mini_desc_path, 'w') as f:
        for image_id in features.keys():
            if image_id in descriptions:
                for desc in descriptions[image_id]:
                    # Retirer les marqueurs de début et de fin
                    clean_desc = desc.replace('startseq ', '').replace(' endseq', '')
                    f.write(f"{image_id}.jpg {clean_desc}\n")
    
    # Créer un fichier d'ensemble d'entraînement minimal
    mini_train_path = os.path.join(text_dir, 'Flickr_8k.trainImages.txt')
    with open(mini_train_path, 'w') as f:
        for image_id in features.keys():
            f.write(f"{image_id}.jpg\n")
    
    # Copier également comme fichiers de validation et de test
    shutil.copy2(mini_train_path, os.path.join(text_dir, 'Flickr_8k.devImages.txt'))
    shutil.copy2(mini_train_path, os.path.join(text_dir, 'Flickr_8k.testImages.txt'))
    
    # Sauvegarder les caractéristiques
    train_features_path = os.path.join(features_dir, 'train_features.npy')
    val_features_path = os.path.join(features_dir, 'val_features.npy')
    test_features_path = os.path.join(features_dir, 'test_features.npy')
    
    np.save(train_features_path, features)
    np.save(val_features_path, features)
    np.save(test_features_path, features)
    
    print(f"\n=== ENSEMBLE DE DONNÉES MINIMAL CRÉÉ AVEC SUCCÈS ===")
    print(f"Images: {len(features)}")
    print(f"Répertoire d'images: {images_dir}")
    print(f"Répertoire de textes: {text_dir}")
    print(f"Répertoire de caractéristiques: {features_dir}")
    
    # Créer le tokenizer
    all_desc = []
    for image_id in features.keys():
        if image_id in descriptions:
            for desc in descriptions[image_id]:
                all_desc.append(desc)
    
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_desc)
    
    # Sauvegarder le tokenizer
    tokenizer_path = os.path.join(model_dir, 'tokenizer.pkl')
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)
    
    print(f"Tokenizer créé et sauvegardé: {tokenizer_path}")
    print(f"Taille du vocabulaire: {len(tokenizer.word_index) + 1}")
    
    return minimal_dir, images_dir, text_dir, features_dir

def main():
    """
    Fonction principale pour créer un ensemble de données minimal.
    """
    print("=== CRÉATION D'UN ENSEMBLE DE DONNÉES COHÉRENT ===")
    
    # Définir le répertoire de base
    base_dir = os.path.abspath(os.path.dirname(__file__))
    
    # Créer l'ensemble de données minimal
    minimal_dir, images_dir, text_dir, features_dir = create_minimal_dataset(base_dir, num_images=20)
    
    if minimal_dir:
        print("\n=== INSTRUCTIONS POUR L'ENTRAÎNEMENT ===")
        print("Pour entraîner le modèle avec cet ensemble minimal, exécutez:")
        print("python src/training/train_simplified.py")
        print("\nLe script utilisera automatiquement l'ensemble minimal créé.")
    else:
        print("\nÉCHEC: Impossible de créer l'ensemble de données minimal.")

if __name__ == "__main__":
    main()
