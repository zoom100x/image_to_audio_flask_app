"""
Script d'entraînement simplifié pour le modèle de génération de légendes.
Version finale utilisant des données en mémoire au lieu de générateurs.
"""
import os
import sys
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Ajouter le répertoire parent au chemin pour pouvoir importer les modules du projet
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.caption_model import define_model
from src.training.trainer_final_memory import setup_callbacks, train_model, prepare_data_in_memory

def load_descriptions(filename):
    """
    Charge et parse les descriptions des images.
    
    Args:
        filename (str): Chemin vers le fichier de descriptions
        
    Returns:
        dict: Dictionnaire des descriptions par image_id
    """
    # Ouvrir le fichier
    file = open(filename, 'r')
    # Lire toutes les lignes
    doc = file.read()
    file.close()
    
    # Traiter chaque ligne
    mapping = {}
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
        
        # Créer la liste si nécessaire
        if image_id not in mapping:
            mapping[image_id] = []
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

def create_tokenizer(descriptions):
    """
    Crée un tokenizer à partir des descriptions.
    
    Args:
        descriptions (dict): Dictionnaire des descriptions
        
    Returns:
        Tokenizer: Tokenizer entraîné
    """
    # Collecter toutes les descriptions
    all_desc = []
    for key in descriptions.keys():
        for desc in descriptions[key]:
            all_desc.append(desc)
    
    # Créer le tokenizer
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_desc)
    
    return tokenizer

def max_length(descriptions):
    """
    Calcule la longueur maximale des descriptions.
    
    Args:
        descriptions (dict): Dictionnaire des descriptions
        
    Returns:
        int: Longueur maximale
    """
    all_desc = []
    for key in descriptions.keys():
        for desc in descriptions[key]:
            all_desc.append(desc)
    
    return max(len(d.split()) for d in all_desc)

def main():
    """
    Fonction principale pour l'entraînement du modèle.
    """
    print("Début de l'entraînement...")
    
    # Définir les chemins (utiliser l'ensemble minimal créé par le diagnostic)
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    data_dir = os.path.join(base_dir, "data")
    
    # Utiliser l'ensemble minimal si disponible, sinon utiliser les chemins standard
    minimal_dir = os.path.join(data_dir, "minimal")
    if os.path.exists(minimal_dir):
        print("Utilisation de l'ensemble de données minimal...")
        image_dir = os.path.join(minimal_dir, "images")
        text_dir = os.path.join(minimal_dir, "text")
        features_dir = os.path.join(minimal_dir, "features")
    else:
        print("Ensemble minimal non trouvé, utilisation des chemins standard...")
        image_dir = os.path.join(data_dir, "flickr8k/Flickr8k_Dataset")
        text_dir = os.path.join(data_dir, "flickr8k/Flickr8k_text")
        features_dir = os.path.join(data_dir, "features")
    
    # Créer les répertoires de sortie
    model_dir = os.path.join(data_dir, "models")
    log_dir = os.path.join(data_dir, "logs")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Charger les descriptions
    descriptions_path = os.path.join(text_dir, 'Flickr8k.token.txt')
    descriptions = load_descriptions(descriptions_path)
    
    # Charger les ensembles d'entraînement, validation et test
    train_path = os.path.join(text_dir, 'Flickr_8k.trainImages.txt')
    val_path = os.path.join(text_dir, 'Flickr_8k.devImages.txt')
    train_set = load_set(train_path)
    val_set = load_set(val_path)
    
    # Filtrer les descriptions pour chaque ensemble
    train_descriptions = {k: v for k, v in descriptions.items() if k in train_set}
    val_descriptions = {k: v for k, v in descriptions.items() if k in val_set}
    
    print(f"Descriptions d'entraînement: {len(train_descriptions)}")
    print(f"Descriptions de validation: {len(val_descriptions)}")
    
    # Charger les caractéristiques
    train_features_path = os.path.join(features_dir, 'train_features.npy')
    val_features_path = os.path.join(features_dir, 'val_features.npy')
    
    try:
        train_features = np.load(train_features_path, allow_pickle=True).item()
        val_features = np.load(val_features_path, allow_pickle=True).item()
        
        print(f"Caractéristiques d'entraînement: {len(train_features)}")
        print(f"Caractéristiques de validation: {len(val_features)}")
    except Exception as e:
        print(f"Erreur lors du chargement des caractéristiques: {str(e)}")
        print("Exécutez d'abord le script de création de dataset synthétique.")
        return
    
    # Filtrer les descriptions pour ne garder que celles qui ont des caractéristiques
    train_descriptions = {k: v for k, v in train_descriptions.items() if k in train_features}
    val_descriptions = {k: v for k, v in val_descriptions.items() if k in val_features}
    
    print(f"Descriptions d'entraînement après filtrage: {len(train_descriptions)}")
    print(f"Descriptions de validation après filtrage: {len(val_descriptions)}")
    
    # Vérifier s'il y a suffisamment d'images pour l'entraînement
    if len(train_descriptions) < 1 or len(val_descriptions) < 1:
        print("ERREUR: Pas assez d'images valides pour l'entraînement.")
        print("Exécutez d'abord le script de création de dataset synthétique.")
        return
    
    # Créer le tokenizer
    tokenizer = create_tokenizer(train_descriptions)
    vocab_size = len(tokenizer.word_index) + 1
    print(f"Taille du vocabulaire: {vocab_size}")
    
    # Déterminer la longueur maximale
    max_seq_length = max_length(train_descriptions)
    print(f"Longueur maximale des séquences: {max_seq_length}")
    
    # Sauvegarder le tokenizer
    tokenizer_path = os.path.join(model_dir, 'tokenizer.pkl')
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)
    
    # Définir le modèle
    model = define_model(vocab_size, max_seq_length)
    
    # Configurer les callbacks
    checkpoint_path = os.path.join(model_dir, 'best_model.weights.h5')
    callbacks = setup_callbacks(checkpoint_path, log_dir)
    
    # Entraîner le modèle avec l'approche en mémoire
    history, trained_model = train_model(
        model,
        train_descriptions,
        train_features,
        val_descriptions,
        val_features,
        tokenizer,
        max_seq_length,
        callbacks,
        epochs=10,
        batch_size=32
    )
    
    if history is not None:
        # Sauvegarder le modèle final
        final_model_path = os.path.join(model_dir, 'final_model.h5')
        trained_model.save(final_model_path)
        print(f"Modèle final sauvegardé dans {final_model_path}")
        
        print("Entraînement terminé avec succès!")
    else:
        print("Erreur lors de l'entraînement du modèle.")

if __name__ == "__main__":
    main()
