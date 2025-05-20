"""
Module pour l'entraînement du modèle de génération de légendes.
Version finale utilisant des données en mémoire au lieu de générateurs.
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.preprocessing.sequence import pad_sequences

def setup_callbacks(checkpoint_path, log_dir):
    """
    Configure les callbacks pour l'entraînement.
    
    Args:
        checkpoint_path (str): Chemin pour sauvegarder les checkpoints
        log_dir (str): Répertoire pour les logs TensorBoard
        
    Returns:
        list: Liste des callbacks configurés
    """
    # Créer les répertoires si nécessaire
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Corriger l'extension du fichier pour save_weights_only=True
    weights_path = os.path.splitext(checkpoint_path)[0] + '.weights.h5'
    
    # Callback pour sauvegarder le meilleur modèle
    checkpoint = ModelCheckpoint(
        weights_path,
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        save_weights_only=True
    )
    
    # Callback pour arrêter l'entraînement si pas d'amélioration
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        verbose=1,
        restore_best_weights=True
    )
    
    # Callback pour réduire le taux d'apprentissage
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        verbose=1,
        min_lr=1e-6
    )
    
    # Callback pour TensorBoard
    tensorboard = TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        write_graph=True
    )
    
    return [checkpoint, early_stopping, reduce_lr, tensorboard]

def prepare_data_in_memory(descriptions, features, tokenizer, max_length):
    """
    Prépare les données d'entraînement en mémoire.
    
    Args:
        descriptions (dict): Dictionnaire des descriptions
        features (dict): Caractéristiques des images
        tokenizer (Tokenizer): Tokenizer entraîné
        max_length (int): Longueur maximale des séquences
        
    Returns:
        tuple: Entrées (X1, X2) et sorties (y) pour l'entraînement
    """
    # Filtrer les descriptions pour ne garder que celles qui ont des caractéristiques
    valid_descriptions = {k: v for k, v in descriptions.items() if k in features}
    
    # Vérifier s'il y a des descriptions valides
    if not valid_descriptions:
        print("ERREUR: Aucune description valide trouvée pour l'entraînement.")
        return None, None, None
    
    print(f"Nombre d'images valides pour la préparation des données: {len(valid_descriptions)}")
    
    X1, X2, y = [], [], []
    
    # Pour chaque image
    for image_id, desc_list in valid_descriptions.items():
        # Pour chaque description
        for desc in desc_list:
            # Encoder la séquence
            seq = tokenizer.texts_to_sequences([desc])[0]
            
            # Diviser en entrées et sorties
            for i in range(1, len(seq)):
                # Séquence d'entrée
                in_seq = seq[:i]
                # Séquence de sortie
                out_seq = seq[i]
                
                # Padding de la séquence d'entrée
                in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                
                # One-hot encoding de la sortie
                out_seq = tf.keras.utils.to_categorical([out_seq], num_classes=len(tokenizer.word_index) + 1)[0]
                
                # Caractéristiques de l'image
                image_feature = features[image_id]
                
                # Ajouter aux listes
                X1.append(image_feature)
                X2.append(in_seq)
                y.append(out_seq)
    
    # Convertir en tableaux numpy
    X1 = np.array(X1)
    X2 = np.array(X2)
    y = np.array(y)
    
    print(f"Données préparées: X1 shape={X1.shape}, X2 shape={X2.shape}, y shape={y.shape}")
    
    return X1, X2, y

def train_model(model, train_descriptions, train_features, val_descriptions, val_features, tokenizer, max_length, callbacks, epochs=10, batch_size=32):
    """
    Entraîne le modèle avec les données en mémoire.
    
    Args:
        model: Modèle à entraîner
        train_descriptions (dict): Descriptions d'entraînement
        train_features (dict): Caractéristiques d'entraînement
        val_descriptions (dict): Descriptions de validation
        val_features (dict): Caractéristiques de validation
        tokenizer (Tokenizer): Tokenizer entraîné
        max_length (int): Longueur maximale des séquences
        callbacks (list): Liste des callbacks
        epochs (int): Nombre d'époques
        batch_size (int): Taille des lots
        
    Returns:
        tuple: Historique d'entraînement et modèle entraîné
    """
    # Préparer les données d'entraînement
    print("Préparation des données d'entraînement...")
    train_X1, train_X2, train_y = prepare_data_in_memory(train_descriptions, train_features, tokenizer, max_length)
    
    # Préparer les données de validation
    print("Préparation des données de validation...")
    val_X1, val_X2, val_y = prepare_data_in_memory(val_descriptions, val_features, tokenizer, max_length)
    
    # Vérifier si les données sont valides
    if train_X1 is None or val_X1 is None:
        print("ERREUR: Impossible de préparer les données.")
        return None, model
    
    # Entraîner le modèle
    print("Début de l'entraînement...")
    history = model.fit(
        [train_X1, train_X2], train_y,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=([val_X1, val_X2], val_y),
        callbacks=callbacks,
        verbose=1
    )
    
    return history, model
