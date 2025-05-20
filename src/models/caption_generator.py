"""
Module pour la génération de légendes à partir des caractéristiques d'une image.
"""
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

def generate_caption(model, tokenizer, photo, max_length):
    """
    Génère une légende pour une image à partir de ses caractéristiques.
    
    Args:
        model: Modèle entraîné
        tokenizer: Tokenizer utilisé pour l'entraînement
        photo: Caractéristiques de l'image
        max_length (int): Longueur maximale de la légende
        
    Returns:
        str: Légende générée
    """
    # Initialiser la séquence avec le jeton de début
    in_text = 'startseq'
    
    # Générer la légende mot par mot
    for i in range(max_length):
        # Encoder la séquence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # Padding de la séquence
        sequence = pad_sequences([sequence], maxlen=max_length)
        
        # Prédire le prochain mot
        yhat = model.predict([photo.reshape(1, -1), sequence], verbose=0)
        # Obtenir l'indice du mot avec la probabilité la plus élevée
        yhat = np.argmax(yhat)
        # Convertir l'indice en mot
        word = word_for_id(yhat, tokenizer)
        
        # Arrêter si le jeton de fin est prédit
        if word is None or word == 'endseq':
            break
            
        # Ajouter le mot à la séquence
        in_text += ' ' + word
    
    # Nettoyer la légende générée
    caption = in_text.replace('startseq', '').strip()
    
    return caption

def word_for_id(word_id, tokenizer):
    """
    Convertit un ID de mot en mot.
    
    Args:
        word_id (int): ID du mot
        tokenizer: Tokenizer utilisé pour l'entraînement
        
    Returns:
        str: Mot correspondant à l'ID
    """
    # Inverser le dictionnaire word_index
    for word, index in tokenizer.word_index.items():
        if index == word_id:
            return word
    return None
