"""
Module pour l'entraînement du modèle de génération de légendes.
"""
import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard

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
    
    # Callback pour sauvegarder le meilleur modèle
    checkpoint = ModelCheckpoint(
        checkpoint_path,
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

def train_model(model, train_generator, val_generator, epochs, steps_per_epoch, validation_steps, callbacks):
    """
    Entraîne le modèle avec les générateurs de données.
    
    Args:
        model: Modèle à entraîner
        train_generator: Générateur pour les données d'entraînement
        val_generator: Générateur pour les données de validation
        epochs (int): Nombre d'époques
        steps_per_epoch (int): Nombre de pas par époque
        validation_steps (int): Nombre de pas de validation
        callbacks (list): Liste des callbacks
        
    Returns:
        tuple: Historique d'entraînement et modèle entraîné
    """
    # Configuration pour optimiser l'utilisation du M1
    # Utiliser Metal pour l'accélération GPU
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    
    # Entraîner le modèle
    history = model.fit(
        train_generator,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_generator,
        validation_steps=validation_steps,
        callbacks=callbacks,
        workers=8,  # Utiliser tous les cœurs du M1
        use_multiprocessing=True
    )
    
    return history, model
