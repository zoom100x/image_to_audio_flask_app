"""
Utilitaires pour l'API Flask.
Version corrigée pour résoudre le problème de chargement du modèle avec NotEqual layer.
"""
import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from src.models.caption_generator import generate_caption
from src.utils.audio_utils import text_to_speech_with_fallback

# Définir les objets personnalisés pour le chargement du modèle
# Cela résout le problème "Unknown layer: 'NotEqual'"
class NotEqualLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(NotEqualLayer, self).__init__(**kwargs)
    
    def call(self, inputs):
        return tf.not_equal(inputs[0], inputs[1])

# Dictionnaire des objets personnalisés
CUSTOM_OBJECTS = {
    'NotEqual': NotEqualLayer,
}

class ImageToAudioPipeline:
    """
    Pipeline pour convertir une image en audio.
    """
    def __init__(self, model_dir='data/models'):
        """
        Initialise le pipeline.
        
        Args:
            model_dir (str): Répertoire contenant le modèle et le tokenizer
        """
        self.model_dir = model_dir
        self.model_path = os.path.join(model_dir, 'final_model.h5')
        self.tokenizer_path = os.path.join(model_dir, 'tokenizer.pkl')
        self.max_length = 30
        
        # Charger le modèle et le tokenizer
        self.load_model_and_tokenizer()
        
        # Charger le modèle d'extraction de caractéristiques
        self.feature_extractor = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
        
        print("Pipeline initialisé avec succès!")
    
    def load_model_and_tokenizer(self):
        """
        Charge le modèle et le tokenizer.
        """
        try:
            # Essayer de charger le modèle avec les objets personnalisés
            print("Tentative de chargement du modèle avec objets personnalisés...")
            self.model = load_model(self.model_path, custom_objects=CUSTOM_OBJECTS)
            print("Modèle chargé avec succès!")
        except Exception as e:
            print(f"Erreur lors du chargement du modèle: {str(e)}")
            print("Tentative de recréation du modèle...")
            
            # Si le chargement échoue, recréer le modèle et charger les poids
            from src.models.caption_model import define_model
            
            # Charger le tokenizer pour obtenir la taille du vocabulaire
            with open(self.tokenizer_path, 'rb') as f:
                self.tokenizer = pickle.load(f)
            
            vocab_size = len(self.tokenizer.word_index) + 1
            
            # Recréer le modèle
            self.model = define_model(vocab_size, self.max_length)
            
            # Essayer de charger les poids
            try:
                # Essayer d'abord avec le chemin direct
                self.model.load_weights(self.model_path)
                print("Poids chargés avec succès!")
            except Exception as e1:
                # Si cela échoue, essayer avec l'extension .weights.h5
                try:
                    weights_path = os.path.splitext(self.model_path)[0] + '.weights.h5'
                    self.model.load_weights(weights_path)
                    print(f"Poids chargés depuis {weights_path}")
                except Exception as e2:
                    print(f"Erreur lors du chargement des poids: {str(e2)}")
                    print("Impossible de charger le modèle ou les poids.")
                    raise e2
            return
        
        # Charger le tokenizer
        try:
            with open(self.tokenizer_path, 'rb') as f:
                self.tokenizer = pickle.load(f)
            print("Tokenizer chargé avec succès!")
        except Exception as e:
            print(f"Erreur lors du chargement du tokenizer: {str(e)}")
            raise e
    
    def extract_features(self, image_path):
        """
        Extrait les caractéristiques d'une image.
        
        Args:
            image_path (str): Chemin vers l'image
            
        Returns:
            numpy.ndarray: Caractéristiques extraites
        """
        # Charger et prétraiter l'image
        img = load_img(image_path, target_size=(299, 299))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        # Extraire les caractéristiques
        feature = self.feature_extractor.predict(img_array, verbose=0)
        
        return feature[0]
    
    # Définir les chemins
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    output_dir = os.path.join(base_dir, "output")
    
    def process_image(self, image_path, output_dir=('static/audio'), lang='fr'):
        """
        Traite une image pour générer une légende et la convertir en audio.
        
        Args:
            image_path (str): Chemin vers l'image
            output_dir (str): Répertoire de sortie pour l'audio
            lang (str): Langue pour la synthèse vocale
            
        Returns:
            tuple: Légende générée et chemin vers le fichier audio
        """
        # Créer le répertoire de sortie si nécessaire
        os.makedirs(output_dir, exist_ok=True)
        
        # Extraire les caractéristiques de l'image
        features = self.extract_features(image_path)
        
        # Générer la légende
        caption = generate_caption(self.model, self.tokenizer, features, self.max_length)
        print(f"Légende générée : {caption}")
        
        # Convertir en audio
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        audio_path = os.path.join(output_dir, f"{image_name}.mp3")
        text_to_speech_with_fallback(caption, audio_path, lang=lang)
        print(f"Audio généré : {audio_path}")
        
        return caption, audio_path
