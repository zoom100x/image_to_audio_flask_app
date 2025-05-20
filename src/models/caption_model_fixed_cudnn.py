"""
Module pour la définition du modèle de génération de légendes.
Version optimisée pour MacBook M1 avec désactivation de cuDNN pour LSTM.
"""
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, Add

def define_model(vocab_size, max_length):
    """
    Définit le modèle de génération de légendes.
    
    Args:
        vocab_size (int): Taille du vocabulaire
        max_length (int): Longueur maximale des séquences
        
    Returns:
        Model: Modèle défini
    """
    # Caractéristiques de l'image (vecteur de 2048 éléments)
    inputs1 = Input(shape=(2048,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    
    # Séquence de mots (entrée textuelle)
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    
    # Utiliser LSTM avec cuDNN désactivé pour éviter les problèmes de masquage
    # L'option use_cudnn=False est implicitement définie par unroll=True
    se3 = LSTM(256, unroll=True, recurrent_activation='sigmoid')(se2)
    
    # Fusionner les deux entrées
    decoder1 = Add()([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    
    # Connecter les entrées et sorties
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    
    # Compiler le modèle
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam'
    )
    
    # Résumé du modèle
    print(model.summary())
    
    return model

def load_model_weights(model, weights_path):
    """
    Charge les poids d'un modèle.
    
    Args:
        model (Model): Modèle à charger
        weights_path (str): Chemin vers les poids
        
    Returns:
        Model: Modèle avec les poids chargés
    """
    try:
        model.load_weights(weights_path)
        print(f"Poids chargés depuis {weights_path}")
    except Exception as e:
        print(f"Erreur lors du chargement des poids: {str(e)}")
    
    return model
