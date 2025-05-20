"""
Module pour l'inférence avec le modèle entraîné.
Ce script permet de générer des légendes pour de nouvelles images et de les convertir en audio.
"""
import os
import sys
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Ajouter le répertoire parent au chemin pour pouvoir importer les modules du projet
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.caption_generator import generate_caption
from src.utils.audio_utils import text_to_speech_with_fallback

def extract_features(image_path):
    """
    Extrait les caractéristiques d'une image à l'aide d'InceptionV3.
    
    Args:
        image_path (str): Chemin vers l'image
        
    Returns:
        numpy.ndarray: Caractéristiques extraites
    """
    # Charger le modèle InceptionV3 pré-entraîné sans la couche de classification
    model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
    
    # Charger et prétraiter l'image
    img = load_img(image_path, target_size=(299, 299))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    # Extraire les caractéristiques
    feature = model.predict(img_array, verbose=0)
    
    return feature[0]

def load_model_and_tokenizer(model_path, tokenizer_path):
    """
    Charge le modèle et le tokenizer.
    
    Args:
        model_path (str): Chemin vers le modèle
        tokenizer_path (str): Chemin vers le tokenizer
        
    Returns:
        tuple: Modèle et tokenizer chargés
    """
    # Charger le modèle
    model = load_model(model_path)
    
    # Charger le tokenizer
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    
    return model, tokenizer

def process_image(image_path, model, tokenizer, max_length=30, output_dir='output', lang='fr'):
    """
    Traite une image pour générer une légende et la convertir en audio.
    
    Args:
        image_path (str): Chemin vers l'image
        model: Modèle entraîné
        tokenizer: Tokenizer utilisé pour l'entraînement
        max_length (int): Longueur maximale de la légende
        output_dir (str): Répertoire de sortie pour l'audio
        lang (str): Langue pour la synthèse vocale
        
    Returns:
        tuple: Légende générée et chemin vers le fichier audio
    """
    # Créer le répertoire de sortie si nécessaire
    os.makedirs(output_dir, exist_ok=True)
    
    # Extraire les caractéristiques de l'image
    features = extract_features(image_path)
    
    # Générer la légende
    caption = generate_caption(model, tokenizer, features, max_length)
    print(f"Légende générée : {caption}")
    
    # Convertir en audio
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    audio_path = os.path.join(output_dir, f"{image_name}.mp3")
    text_to_speech_with_fallback(caption, audio_path, lang=lang)
    print(f"Audio généré : {audio_path}")
    
    return caption, audio_path

def main():
    """
    Fonction principale pour l'inférence.
    """
    # Vérifier les arguments
    if len(sys.argv) < 2:
        print("Usage: python inference.py <chemin_image>")
        return
    
    # Chemin vers l'image
    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"Erreur: L'image {image_path} n'existe pas.")
        return
    
    # Définir les chemins
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    model_dir = os.path.join(base_dir, "data/models")
    output_dir = os.path.join(base_dir, "output")
    
    # Chemins vers le modèle et le tokenizer
    model_path = os.path.join(model_dir, "final_model.h5")
    tokenizer_path = os.path.join(model_dir, "tokenizer.pkl")
    
    # Vérifier si les fichiers existent
    if not os.path.exists(model_path):
        print(f"Erreur: Le modèle {model_path} n'existe pas.")
        return
    
    if not os.path.exists(tokenizer_path):
        print(f"Erreur: Le tokenizer {tokenizer_path} n'existe pas.")
        return
    
    # Charger le modèle et le tokenizer
    model, tokenizer = load_model_and_tokenizer(model_path, tokenizer_path)
    
    # Traiter l'image
    caption, audio_path = process_image(image_path, model, tokenizer, output_dir=output_dir)
    
    print("\nTraitement terminé avec succès!")
    print(f"Image: {image_path}")
    print(f"Légende: {caption}")
    print(f"Audio: {audio_path}")

if __name__ == "__main__":
    main()
