"""
Script de diagnostic pour vérifier et corriger la cohérence des données.
Ce script vérifie la correspondance entre les images, les descriptions et les caractéristiques extraites.
"""
import os
import sys
import numpy as np
import shutil
from collections import defaultdict
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Ajouter le répertoire parent au chemin pour pouvoir importer les modules du projet
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def check_file_exists(filepath, description):
    """Vérifie si un fichier existe et affiche un message approprié."""
    if os.path.exists(filepath):
        print(f"✓ {description} trouvé: {filepath}")
        return True
    else:
        print(f"✗ {description} manquant: {filepath}")
        return False

def check_directory_exists(dirpath, description):
    """Vérifie si un répertoire existe et affiche un message approprié."""
    if os.path.exists(dirpath) and os.path.isdir(dirpath):
        print(f"✓ {description} trouvé: {dirpath}")
        return True
    else:
        print(f"✗ {description} manquant: {dirpath}")
        return False

def load_descriptions(filename):
    """
    Charge et parse les descriptions des images.
    
    Args:
        filename (str): Chemin vers le fichier de descriptions
        
    Returns:
        dict: Dictionnaire des descriptions par image_id
    """
    if not check_file_exists(filename, "Fichier de descriptions"):
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

def load_set(filename):
    """
    Charge les identifiants des images pour un ensemble spécifique (train, val, test).
    
    Args:
        filename (str): Chemin vers le fichier d'identifiants
        
    Returns:
        set: Ensemble d'identifiants d'images
    """
    if not check_file_exists(filename, f"Fichier d'ensemble {os.path.basename(filename)}"):
        return set()
        
    # Ouvrir le fichier
    file = open(filename, 'r')
    # Lire toutes les lignes
    doc = file.read()
    file.close()
    
    # Traiter chaque ligne
    dataset = []
    for line in doc.split('\n'):
        # Ignorer les lignes vides
        if len(line) < 1:
            continue
        # Extraire l'identifiant de l'image
        identifier = line.split('.')[0]
        dataset.append(identifier)
    
    print(f"Nombre d'identifiants dans {os.path.basename(filename)}: {len(dataset)}")
    return set(dataset)

def check_image_files(image_dir, image_ids):
    """
    Vérifie si les fichiers d'images existent pour les identifiants donnés.
    
    Args:
        image_dir (str): Répertoire contenant les images
        image_ids (set): Ensemble d'identifiants d'images
        
    Returns:
        set: Ensemble d'identifiants d'images existantes
    """
    if not check_directory_exists(image_dir, "Répertoire d'images"):
        return set()
        
    existing_images = set()
    missing_images = set()
    
    for image_id in image_ids:
        image_path = os.path.join(image_dir, f"{image_id}.jpg")
        if os.path.exists(image_path):
            existing_images.add(image_id)
        else:
            missing_images.add(image_id)
    
    print(f"Images existantes: {len(existing_images)}/{len(image_ids)}")
    if missing_images:
        print(f"Images manquantes: {len(missing_images)}")
        if len(missing_images) < 10:
            print(f"Exemples d'images manquantes: {list(missing_images)}")
        else:
            print(f"Exemples d'images manquantes: {list(missing_images)[:10]}...")
    
    return existing_images

def extract_features_for_subset(image_dir, image_ids, max_images=100):
    """
    Extrait les caractéristiques pour un sous-ensemble d'images.
    
    Args:
        image_dir (str): Répertoire contenant les images
        image_ids (set): Ensemble d'identifiants d'images
        max_images (int): Nombre maximum d'images à traiter
        
    Returns:
        dict: Caractéristiques extraites par image_id
    """
    if not image_ids:
        print("Aucune image à traiter.")
        return {}
        
    # Limiter le nombre d'images pour le diagnostic
    subset_ids = list(image_ids)[:max_images]
    print(f"Extraction des caractéristiques pour {len(subset_ids)} images...")
    
    try:
        # Charger le modèle InceptionV3 pré-entraîné sans la couche de classification
        model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
        
        # Dictionnaire pour stocker les caractéristiques
        features = {}
        
        # Traiter les images une par une
        for i, image_id in enumerate(subset_ids):
            image_path = os.path.join(image_dir, f"{image_id}.jpg")
            if os.path.exists(image_path):
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
                    if (i + 1) % 10 == 0:
                        print(f"Traitement: {i + 1}/{len(subset_ids)}")
                except Exception as e:
                    print(f"Erreur lors du traitement de l'image {image_id}: {str(e)}")
        
        print(f"Caractéristiques extraites pour {len(features)} images.")
        return features
    
    except Exception as e:
        print(f"Erreur lors de l'extraction des caractéristiques: {str(e)}")
        return {}

def save_features(features, filename):
    """
    Sauvegarde les caractéristiques extraites.
    
    Args:
        features (dict): Caractéristiques à sauvegarder
        filename (str): Chemin du fichier de sortie
    """
    if not features:
        print("Aucune caractéristique à sauvegarder.")
        return False
        
    try:
        # Créer le répertoire si nécessaire
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Sauvegarder les caractéristiques
        np.save(filename, features)
        print(f"Caractéristiques sauvegardées dans {filename}")
        return True
    except Exception as e:
        print(f"Erreur lors de la sauvegarde des caractéristiques: {str(e)}")
        return False

def load_features(filename):
    """
    Charge les caractéristiques extraites.
    
    Args:
        filename (str): Chemin du fichier de caractéristiques
        
    Returns:
        dict: Caractéristiques chargées
    """
    if not check_file_exists(filename, "Fichier de caractéristiques"):
        return {}
        
    try:
        features = np.load(filename, allow_pickle=True).item()
        print(f"Caractéristiques chargées: {len(features)}")
        return features
    except Exception as e:
        print(f"Erreur lors du chargement des caractéristiques: {str(e)}")
        return {}

def check_features_descriptions_match(features, descriptions):
    """
    Vérifie la correspondance entre les caractéristiques et les descriptions.
    
    Args:
        features (dict): Caractéristiques extraites
        descriptions (dict): Descriptions des images
        
    Returns:
        set: Ensemble d'identifiants d'images communs
    """
    feature_ids = set(features.keys())
    description_ids = set(descriptions.keys())
    
    common_ids = feature_ids & description_ids
    
    print(f"Identifiants dans les caractéristiques: {len(feature_ids)}")
    print(f"Identifiants dans les descriptions: {len(description_ids)}")
    print(f"Identifiants communs: {len(common_ids)}")
    
    if len(common_ids) == 0:
        print("ERREUR CRITIQUE: Aucune correspondance entre les caractéristiques et les descriptions!")
        print("Exemples d'identifiants de caractéristiques:", list(feature_ids)[:5])
        print("Exemples d'identifiants de descriptions:", list(description_ids)[:5])
    
    return common_ids

def create_minimal_dataset(image_dir, text_dir, features_dir, output_dir, max_images=100):
    """
    Crée un ensemble de données minimal pour le diagnostic.
    
    Args:
        image_dir (str): Répertoire contenant les images
        text_dir (str): Répertoire contenant les fichiers texte
        features_dir (str): Répertoire pour les caractéristiques
        output_dir (str): Répertoire de sortie
        max_images (int): Nombre maximum d'images
        
    Returns:
        tuple: Chemins vers les répertoires créés
    """
    # Créer les répertoires de sortie
    mini_image_dir = os.path.join(output_dir, "images")
    mini_text_dir = os.path.join(output_dir, "text")
    mini_features_dir = os.path.join(output_dir, "features")
    
    os.makedirs(mini_image_dir, exist_ok=True)
    os.makedirs(mini_text_dir, exist_ok=True)
    os.makedirs(mini_features_dir, exist_ok=True)
    
    # Charger les descriptions
    descriptions_path = os.path.join(text_dir, 'Flickr8k.token.txt')
    descriptions = load_descriptions(descriptions_path)
    
    # Charger les ensembles d'entraînement
    train_path = os.path.join(text_dir, 'Flickr_8k.trainImages.txt')
    train_set = load_set(train_path)
    
    # Vérifier les fichiers d'images
    existing_images = check_image_files(image_dir, train_set)
    
    # Sélectionner un sous-ensemble d'images
    subset_ids = list(existing_images)[:max_images]
    if not subset_ids:
        print("Aucune image valide trouvée.")
        return None, None, None
    
    print(f"Création d'un ensemble minimal avec {len(subset_ids)} images...")
    
    # Copier les images
    for image_id in subset_ids:
        src_path = os.path.join(image_dir, f"{image_id}.jpg")
        dst_path = os.path.join(mini_image_dir, f"{image_id}.jpg")
        shutil.copy2(src_path, dst_path)
    
    # Créer un fichier de descriptions minimal
    mini_desc_path = os.path.join(mini_text_dir, 'Flickr8k.token.txt')
    with open(mini_desc_path, 'w') as f:
        for image_id in subset_ids:
            if image_id in descriptions:
                for desc in descriptions[image_id]:
                    # Retirer les marqueurs de début et de fin
                    clean_desc = desc.replace('startseq ', '').replace(' endseq', '')
                    f.write(f"{image_id}.jpg {clean_desc}\n")
    
    # Créer un fichier d'ensemble d'entraînement minimal
    mini_train_path = os.path.join(mini_text_dir, 'Flickr_8k.trainImages.txt')
    with open(mini_train_path, 'w') as f:
        for image_id in subset_ids:
            f.write(f"{image_id}.jpg\n")
    
    # Copier également comme fichiers de validation et de test
    shutil.copy2(mini_train_path, os.path.join(mini_text_dir, 'Flickr_8k.devImages.txt'))
    shutil.copy2(mini_train_path, os.path.join(mini_text_dir, 'Flickr_8k.testImages.txt'))
    
    # Extraire les caractéristiques
    features = extract_features_for_subset(mini_image_dir, set(subset_ids))
    
    # Sauvegarder les caractéristiques
    if features:
        mini_features_path = os.path.join(mini_features_dir, 'train_features.npy')
        save_features(features, mini_features_path)
        
        # Copier également comme caractéristiques de validation et de test
        shutil.copy2(mini_features_path, os.path.join(mini_features_dir, 'val_features.npy'))
        shutil.copy2(mini_features_path, os.path.join(mini_features_dir, 'test_features.npy'))
    
    print(f"Ensemble minimal créé dans {output_dir}")
    return mini_image_dir, mini_text_dir, mini_features_dir

def run_diagnostic():
    """
    Exécute le diagnostic complet et crée un ensemble minimal.
    """
    print("=== DIAGNOSTIC DU PROJET IMAGE-TO-AUDIO ===")
    
    # Définir les chemins
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    data_dir = os.path.join(base_dir, "data")
    image_dir = os.path.join(data_dir, "flickr8k/Flickr8k_Dataset")
    text_dir = os.path.join(data_dir, "flickr8k/Flickr8k_text")
    features_dir = os.path.join(data_dir, "features")
    output_dir = os.path.join(data_dir, "minimal")
    
    print("\n=== VÉRIFICATION DES RÉPERTOIRES ===")
    check_directory_exists(data_dir, "Répertoire de données")
    check_directory_exists(image_dir, "Répertoire d'images")
    check_directory_exists(text_dir, "Répertoire de textes")
    
    print("\n=== VÉRIFICATION DES FICHIERS DE DESCRIPTION ===")
    descriptions_path = os.path.join(text_dir, 'Flickr8k.token.txt')
    train_path = os.path.join(text_dir, 'Flickr_8k.trainImages.txt')
    val_path = os.path.join(text_dir, 'Flickr_8k.devImages.txt')
    test_path = os.path.join(text_dir, 'Flickr_8k.testImages.txt')
    
    check_file_exists(descriptions_path, "Fichier de descriptions")
    check_file_exists(train_path, "Fichier d'entraînement")
    check_file_exists(val_path, "Fichier de validation")
    check_file_exists(test_path, "Fichier de test")
    
    print("\n=== CHARGEMENT DES DESCRIPTIONS ===")
    descriptions = load_descriptions(descriptions_path)
    
    print("\n=== CHARGEMENT DES ENSEMBLES ===")
    train_set = load_set(train_path)
    val_set = load_set(val_path)
    test_set = load_set(test_path)
    
    print("\n=== VÉRIFICATION DES FICHIERS D'IMAGES ===")
    train_images = check_image_files(image_dir, train_set)
    val_images = check_image_files(image_dir, val_set)
    test_images = check_image_files(image_dir, test_set)
    
    print("\n=== VÉRIFICATION DES CARACTÉRISTIQUES ===")
    train_features_path = os.path.join(features_dir, 'train_features.npy')
    val_features_path = os.path.join(features_dir, 'val_features.npy')
    test_features_path = os.path.join(features_dir, 'test_features.npy')
    
    train_features = load_features(train_features_path)
    val_features = load_features(val_features_path)
    test_features = load_features(test_features_path)
    
    print("\n=== VÉRIFICATION DE LA CORRESPONDANCE ===")
    train_common = check_features_descriptions_match(train_features, {k: v for k, v in descriptions.items() if k in train_set})
    val_common = check_features_descriptions_match(val_features, {k: v for k, v in descriptions.items() if k in val_set})
    test_common = check_features_descriptions_match(test_features, {k: v for k, v in descriptions.items() if k in test_set})
    
    print("\n=== CRÉATION D'UN ENSEMBLE MINIMAL ===")
    mini_image_dir, mini_text_dir, mini_features_dir = create_minimal_dataset(
        image_dir, text_dir, features_dir, output_dir, max_images=50
    )
    
    print("\n=== RÉSUMÉ DU DIAGNOSTIC ===")
    if mini_image_dir and mini_text_dir and mini_features_dir:
        print("✓ Ensemble minimal créé avec succès.")
        print(f"  - Images: {mini_image_dir}")
        print(f"  - Textes: {mini_text_dir}")
        print(f"  - Caractéristiques: {mini_features_dir}")
        print("\nPour utiliser cet ensemble minimal, modifiez les chemins dans train.py:")
        print("  image_dir = 'data/minimal/images'")
        print("  text_dir = 'data/minimal/text'")
        print("  features_dir = 'data/minimal/features'")
    else:
        print("✗ Échec de la création de l'ensemble minimal.")
    
    if len(train_common) > 0 and len(val_common) > 0 and len(test_common) > 0:
        print("\n✓ Des correspondances ont été trouvées entre les caractéristiques et les descriptions.")
        print("  Vous pouvez continuer avec l'entraînement en utilisant les fichiers existants.")
    else:
        print("\n✗ Aucune correspondance trouvée entre les caractéristiques et les descriptions.")
        print("  Utilisez l'ensemble minimal créé pour l'entraînement.")

if __name__ == "__main__":
    run_diagnostic()
