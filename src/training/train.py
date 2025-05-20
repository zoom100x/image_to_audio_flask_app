"""
Script principal d'entraînement du modèle.
"""
import os
import pickle
import numpy as np
import tensorflow as tf
from src.data.dataset import download_flickr8k
from src.data.preprocessing import (
    prepare_datasets, extract_features, save_features,
    create_tokenizer, max_length, data_generator
)
from src.models.caption_model import define_caption_model
from src.training.trainer import setup_callbacks, train_model

def main():
    """
    Script principal d'entraînement.
    """
    # Paramètres
    data_dir = 'data'
    model_dir = os.path.join(data_dir, 'models')
    features_dir = os.path.join(data_dir, 'features')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(features_dir, exist_ok=True)
    
    # Télécharger le dataset
    image_dir, text_dir = download_flickr8k(os.path.join(data_dir, 'flickr8k'))
    
    # Préparer les ensembles de données
    train_descriptions, val_descriptions, test_descriptions = prepare_datasets(text_dir)
    
    # Extraire les caractéristiques des images (si pas déjà fait)
    train_features_path = os.path.join(features_dir, 'train_features.npy')
    val_features_path = os.path.join(features_dir, 'val_features.npy')
    
    if not os.path.exists(train_features_path):
        print("Extraction des caractéristiques des images d'entraînement...")
        train_features = extract_features(image_dir, train_descriptions.keys())
        save_features(train_features, train_features_path)
    else:
        print("Chargement des caractéristiques des images d'entraînement...")
        train_features = np.load(train_features_path, allow_pickle=True).item()
    
    if not os.path.exists(val_features_path):
        print("Extraction des caractéristiques des images de validation...")
        val_features = extract_features(image_dir, val_descriptions.keys())
        save_features(val_features, val_features_path)
    else:
        print("Chargement des caractéristiques des images de validation...")
        val_features = np.load(val_features_path, allow_pickle=True).item()
    
    # Créer le tokenizer
    tokenizer = create_tokenizer(train_descriptions)
    vocab_size = len(tokenizer.word_index) + 1
    print(f"Taille du vocabulaire: {vocab_size}")
    
    # Sauvegarder le tokenizer
    tokenizer_path = os.path.join(model_dir, 'tokenizer.pkl')
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)
    
    # Déterminer la longueur maximale des séquences
    max_seq_length = max_length(train_descriptions)
    print(f"Longueur maximale des séquences: {max_seq_length}")
    
    # Créer les générateurs de données
    train_generator = data_generator(
        train_descriptions, train_features, tokenizer, max_seq_length, batch_size=64
    )
    val_generator = data_generator(
        val_descriptions, val_features, tokenizer, max_seq_length, batch_size=64
    )
    
    # Calculer les pas par époque
    steps_train = len(train_descriptions) // 64
    steps_val = len(val_descriptions) // 64
    
    # Définir le modèle
    model = define_caption_model(vocab_size, max_seq_length)
    print(model.summary())
    
    # Configurer les callbacks
    checkpoint_path = os.path.join(model_dir, 'best_model.h5')
    log_dir = os.path.join(model_dir, 'logs')
    callbacks = setup_callbacks(checkpoint_path, log_dir)
    
    # Entraîner le modèle
    print("Début de l'entraînement...")
    history, trained_model = train_model(
        model, train_generator, val_generator, 
        epochs=20, steps_per_epoch=steps_train, 
        validation_steps=steps_val, callbacks=callbacks
    )
    
    # Sauvegarder le modèle final
    final_model_path = os.path.join(model_dir, 'final_model.h5')
    trained_model.save(final_model_path)
    print(f"Modèle sauvegardé à {final_model_path}")

if __name__ == "__main__":
    main()
