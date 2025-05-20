# Pipeline de Prétraitement et d'Entraînement

Ce document détaille le pipeline complet de prétraitement des données et d'entraînement du modèle pour l'application de conversion image-to-audio, optimisé pour un MacBook Pro M1 avec 16GB de RAM.

## 1. Acquisition et Préparation des Données

### 1.1 Téléchargement du Dataset Flickr8k

```python
# src/data/dataset.py
import os
import requests
import zipfile
import shutil

def download_flickr8k(target_dir='data/flickr8k'):
    """
    Télécharge et extrait le dataset Flickr8k.
    """
    os.makedirs(target_dir, exist_ok=True)
    
    # URLs pour le dataset Flickr8k
    image_url = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip"
    text_url = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip"
    
    # Téléchargement et extraction des images
    image_zip = os.path.join(target_dir, "Flickr8k_Dataset.zip")
    if not os.path.exists(image_zip):
        print("Téléchargement des images Flickr8k...")
        response = requests.get(image_url, stream=True)
        with open(image_zip, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    
    # Extraction des images
    if not os.path.exists(os.path.join(target_dir, "Flickr8k_Dataset")):
        print("Extraction des images...")
        with zipfile.ZipFile(image_zip, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
    
    # Téléchargement et extraction des textes
    text_zip = os.path.join(target_dir, "Flickr8k_text.zip")
    if not os.path.exists(text_zip):
        print("Téléchargement des annotations textuelles...")
        response = requests.get(text_url, stream=True)
        with open(text_zip, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    
    # Extraction des textes
    if not os.path.exists(os.path.join(target_dir, "Flickr8k_text")):
        print("Extraction des annotations...")
        with zipfile.ZipFile(text_zip, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
    
    print("Dataset Flickr8k prêt à l'emploi!")
    return os.path.join(target_dir, "Flickr8k_Dataset"), os.path.join(target_dir, "Flickr8k_text")
```

### 1.2 Chargement et Parsing des Annotations

```python
# src/data/preprocessing.py
import os
import pandas as pd
import numpy as np
from collections import defaultdict

def load_descriptions(filename):
    """
    Charge et parse les descriptions des images.
    """
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
        image_id, image_desc = tokens[0], ' '.join(tokens[1:])
        # Extraire l'identifiant de l'image sans l'extension
        image_id = image_id.split('.')[0]
        # Convertir la description en minuscules
        image_desc = image_desc.lower()
        # Ajouter un marqueur de début et de fin
        image_desc = 'startseq ' + image_desc + ' endseq'
        # Stocker la description
        mapping[image_id].append(image_desc)
    
    return mapping

def load_set(filename):
    """
    Charge les identifiants des images pour un ensemble spécifique (train, val, test).
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
    
    return set(dataset)

def prepare_datasets(text_dir):
    """
    Prépare les ensembles d'entraînement, validation et test.
    """
    # Charger les descriptions
    descriptions_path = os.path.join(text_dir, 'Flickr8k.token.txt')
    descriptions = load_descriptions(descriptions_path)
    
    # Charger les ensembles d'entraînement, validation et test
    train_path = os.path.join(text_dir, 'Flickr_8k.trainImages.txt')
    val_path = os.path.join(text_dir, 'Flickr_8k.devImages.txt')
    test_path = os.path.join(text_dir, 'Flickr_8k.testImages.txt')
    
    train_set = load_set(train_path)
    val_set = load_set(val_path)
    test_set = load_set(test_path)
    
    # Filtrer les descriptions pour chaque ensemble
    train_descriptions = {k: v for k, v in descriptions.items() if k in train_set}
    val_descriptions = {k: v for k, v in descriptions.items() if k in val_set}
    test_descriptions = {k: v for k, v in descriptions.items() if k in test_set}
    
    return train_descriptions, val_descriptions, test_descriptions
```

### 1.3 Prétraitement des Images

```python
# src/data/preprocessing.py (suite)
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def preprocess_image(image_path, target_size=(299, 299)):
    """
    Prétraite une image pour l'extraction de caractéristiques.
    """
    # Charger l'image
    img = load_img(image_path, target_size=target_size)
    # Convertir en tableau
    img_array = img_to_array(img)
    # Redimensionner
    img_array = np.expand_dims(img_array, axis=0)
    # Prétraiter pour le modèle InceptionV3
    img_array = preprocess_input(img_array)
    
    return img_array

def extract_features(image_dir, image_ids, batch_size=32):
    """
    Extrait les caractéristiques des images à l'aide d'InceptionV3.
    Optimisé pour la mémoire avec traitement par lots.
    """
    # Charger le modèle InceptionV3 pré-entraîné sans la couche de classification
    model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
    
    # Dictionnaire pour stocker les caractéristiques
    features = {}
    
    # Traiter les images par lots pour économiser la mémoire
    for i in range(0, len(image_ids), batch_size):
        batch_ids = list(image_ids)[i:i+batch_size]
        print(f"Traitement du lot {i//batch_size + 1}/{len(image_ids)//batch_size + 1}")
        
        # Prétraiter les images du lot
        batch_features = []
        batch_paths = []
        
        for image_id in batch_ids:
            image_path = os.path.join(image_dir, f"{image_id}.jpg")
            if os.path.exists(image_path):
                img_array = preprocess_image(image_path)
                batch_features.append(img_array)
                batch_paths.append(image_id)
        
        if batch_features:
            # Concaténer les images prétraitées
            batch_features = np.vstack(batch_features)
            
            # Extraire les caractéristiques
            batch_output = model.predict(batch_features, verbose=0)
            
            # Stocker les caractéristiques
            for j, image_id in enumerate(batch_paths):
                features[image_id] = batch_output[j]
        
        # Libérer la mémoire
        tf.keras.backend.clear_session()
    
    return features

def save_features(features, filename):
    """
    Sauvegarde les caractéristiques extraites.
    """
    np.save(filename, features)
    print(f"Caractéristiques sauvegardées dans {filename}")
```

### 1.4 Prétraitement du Texte

```python
# src/data/preprocessing.py (suite)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def create_tokenizer(descriptions):
    """
    Crée un tokenizer à partir des descriptions.
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
    """
    all_desc = []
    for key in descriptions.keys():
        for desc in descriptions[key]:
            all_desc.append(desc)
    
    return max(len(d.split()) for d in all_desc)

def create_sequences(tokenizer, max_length, descriptions, features):
    """
    Crée des séquences d'entrée-sortie pour l'entraînement.
    Optimisé pour la mémoire avec génération à la demande.
    """
    X1, X2, y = [], [], []
    
    # Pour chaque image
    for image_id, desc_list in descriptions.items():
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
                out_seq = tf.keras.utils.to_categorical([out_seq], num_classes=tokenizer.num_words)[0]
                
                # Caractéristiques de l'image
                image_feature = features[image_id]
                
                # Ajouter aux listes
                X1.append(image_feature)
                X2.append(in_seq)
                y.append(out_seq)
    
    return np.array(X1), np.array(X2), np.array(y)

def data_generator(descriptions, features, tokenizer, max_length, batch_size=32):
    """
    Générateur de données pour l'entraînement.
    Économise la mémoire en générant les données à la demande.
    """
    # Obtenir les identifiants des images
    image_ids = list(descriptions.keys())
    
    while True:
        # Mélanger les identifiants à chaque époque
        np.random.shuffle(image_ids)
        
        # Créer des lots
        for i in range(0, len(image_ids), batch_size):
            batch_ids = image_ids[i:i+batch_size]
            
            X1, X2, y = [], [], []
            
            # Pour chaque identifiant dans le lot
            for image_id in batch_ids:
                # Pour chaque description
                for desc in descriptions[image_id]:
                    # Encoder la séquence
                    seq = tokenizer.texts_to_sequences([desc])[0]
                    
                    # Diviser en entrées et sorties
                    for j in range(1, len(seq)):
                        # Séquence d'entrée
                        in_seq = seq[:j]
                        # Séquence de sortie
                        out_seq = seq[j]
                        
                        # Padding de la séquence d'entrée
                        in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                        
                        # One-hot encoding de la sortie
                        out_seq = tf.keras.utils.to_categorical([out_seq], num_classes=tokenizer.num_words)[0]
                        
                        # Caractéristiques de l'image
                        image_feature = features[image_id]
                        
                        # Ajouter aux listes
                        X1.append(image_feature)
                        X2.append(in_seq)
                        y.append(out_seq)
            
            yield [np.array(X1), np.array(X2)], np.array(y)
```

## 2. Définition du Modèle

### 2.1 Architecture du Modèle d'Image Captioning

```python
# src/models/caption_model.py
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, Add

def define_caption_model(vocab_size, max_length, embedding_dim=256, lstm_units=256):
    """
    Définit le modèle d'image captioning.
    Architecture: CNN (features) + LSTM pour la génération de texte.
    """
    # Caractéristiques de l'image (sortie de InceptionV3)
    inputs1 = Input(shape=(2048,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(embedding_dim, activation='relu')(fe1)
    
    # Séquence de mots
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(lstm_units)(se2)
    
    # Décodeur
    decoder1 = Add()([fe2, se3])
    decoder2 = Dense(lstm_units, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    
    # Assembler le modèle
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    
    # Compiler le modèle
    # Utiliser mixed_float16 pour optimiser les performances sur M1
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    return model
```

### 2.2 Modèle Alternatif avec Attention (Optionnel)

```python
# src/models/caption_model.py (suite)
from tensorflow.keras.layers import Attention, Concatenate, TimeDistributed, Bidirectional

def define_attention_model(vocab_size, max_length, embedding_dim=256, lstm_units=256):
    """
    Définit un modèle d'image captioning avec mécanisme d'attention.
    Plus performant mais plus gourmand en ressources.
    """
    # Caractéristiques de l'image
    inputs1 = Input(shape=(2048,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(embedding_dim, activation='relu')(fe1)
    fe3 = tf.expand_dims(fe2, axis=1)  # Ajouter une dimension temporelle
    
    # Séquence de mots
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = Bidirectional(LSTM(lstm_units, return_sequences=True))(se2)
    
    # Mécanisme d'attention
    attention = Attention()([se3, fe3])
    
    # Combiner
    decoder1 = Concatenate()([attention, se3])
    decoder2 = TimeDistributed(Dense(lstm_units, activation='relu'))(decoder1)
    decoder3 = LSTM(lstm_units)(decoder2)
    outputs = Dense(vocab_size, activation='softmax')(decoder3)
    
    # Assembler le modèle
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    
    # Compiler le modèle avec mixed_float16 pour M1
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    return model
```

## 3. Entraînement du Modèle

### 3.1 Configuration et Callbacks

```python
# src/training/trainer.py
import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard

def setup_callbacks(checkpoint_path, log_dir):
    """
    Configure les callbacks pour l'entraînement.
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
```

### 3.2 Fonction d'Entraînement

```python
# src/training/trainer.py (suite)
def train_model(model, train_generator, val_generator, epochs, steps_per_epoch, validation_steps, callbacks):
    """
    Entraîne le modèle avec les générateurs de données.
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
```

### 3.3 Script d'Entraînement Principal

```python
# src/training/train.py
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
```

## 4. Génération de Légendes et Conversion en Audio

### 4.1 Génération de Légendes

```python
# src/models/caption_generator.py
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

def generate_caption(model, tokenizer, photo_features, max_length):
    """
    Génère une légende pour une image à partir de ses caractéristiques.
    """
    # Initialiser la séquence avec le token de début
    in_text = 'startseq'
    
    # Générer la légende mot par mot
    for _ in range(max_length):
        # Encoder la séquence partielle
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # Padding
        sequence = pad_sequences([sequence], maxlen=max_length)
        
        # Prédire le prochain mot
        yhat = model.predict([photo_features.reshape(1, -1), sequence], verbose=0)
        # Obtenir l'indice du mot avec la plus haute probabilité
        yhat = np.argmax(yhat)
        # Convertir l'indice en mot
        word = ''
        for w, index in tokenizer.word_index.items():
            if index == yhat:
                word = w
                break
        
        # Arrêter si on atteint le token de fin
        if word == 'endseq':
            break
        
        # Ajouter le mot à la séquence
        in_text += ' ' + word
    
    # Nettoyer la légende (enlever les tokens de début/fin)
    caption = in_text.replace('startseq', '').replace('endseq', '').strip()
    
    return caption
```

### 4.2 Conversion Text-to-Speech

```python
# src/utils/audio_utils.py
import os
from gtts import gTTS
import tempfile

def text_to_speech(text, output_path, lang='fr'):
    """
    Convertit du texte en audio à l'aide de gTTS.
    """
    # Créer le répertoire de sortie si nécessaire
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convertir le texte en audio
    tts = gTTS(text=text, lang=lang, slow=False)
    
    # Sauvegarder l'audio
    tts.save(output_path)
    
    return output_path

def text_to_speech_with_fallback(text, output_path, lang='fr'):
    """
    Convertit du texte en audio avec gestion des erreurs.
    Utilise un fichier temporaire pour éviter les problèmes d'écriture.
    """
    try:
        # Essayer d'abord avec gTTS
        return text_to_speech(text, output_path, lang)
    except Exception as e:
        print(f"Erreur avec gTTS: {e}")
        try:
            # Fallback: utiliser pyttsx3 (TTS local)
            import pyttsx3
            engine = pyttsx3.init()
            
            # Créer un fichier temporaire
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Sauvegarder l'audio dans le fichier temporaire
            engine.save_to_file(text, temp_path)
            engine.runAndWait()
            
            # Copier le fichier temporaire vers la destination finale
            import shutil
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            shutil.copy2(temp_path, output_path)
            
            # Supprimer le fichier temporaire
            os.unlink(temp_path)
            
            return output_path
        except Exception as e2:
            print(f"Erreur avec pyttsx3: {e2}")
            raise Exception("Impossible de convertir le texte en audio.")
```

## 5. Pipeline Complet pour l'Inférence

```python
# src/api/utils.py
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import pickle

from src.models.caption_generator import generate_caption
from src.utils.audio_utils import text_to_speech_with_fallback

class ImageToAudioPipeline:
    """
    Pipeline complet pour la conversion d'image en audio.
    """
    def __init__(self, model_dir='data/models'):
        # Charger le modèle de captioning
        self.model_path = os.path.join(model_dir, 'best_model.h5')
        self.model = load_model(self.model_path)
        
        # Charger le tokenizer
        tokenizer_path = os.path.join(model_dir, 'tokenizer.pkl')
        with open(tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)
        
        # Charger le modèle d'extraction de caractéristiques
        self.feature_extractor = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
        
        # Déterminer la longueur maximale des séquences
        self.max_length = 40  # Valeur par défaut, à ajuster selon votre dataset
        
        # Répertoire pour les sorties audio
        self.audio_dir = 'static/audio'
        os.makedirs(self.audio_dir, exist_ok=True)
    
    def extract_features(self, image_path):
        """
        Extrait les caractéristiques d'une image.
        """
        # Charger et prétraiter l'image
        img = load_img(image_path, target_size=(299, 299))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        # Extraire les caractéristiques
        features = self.feature_extractor.predict(img_array, verbose=0)
        
        return features[0]
    
    def process_image(self, image_path, lang='fr'):
        """
        Traite une image pour générer une légende et un fichier audio.
        """
        # Extraire les caractéristiques
        features = self.extract_features(image_path)
        
        # Générer la légende
        caption = generate_caption(self.model, self.tokenizer, features, self.max_length)
        
        # Générer le nom du fichier audio
        audio_filename = f"audio_{os.path.basename(image_path).split('.')[0]}.mp3"
        audio_path = os.path.join(self.audio_dir, audio_filename)
        
        # Convertir la légende en audio
        audio_path = text_to_speech_with_fallback(caption, audio_path, lang)
        
        return {
            'caption': caption,
            'audio_path': audio_path
        }
```

## 6. Optimisations pour MacBook Pro M1

### 6.1 Configuration TensorFlow pour M1

```python
# src/config.py
import tensorflow as tf
import os

def configure_tensorflow_for_m1():
    """
    Configure TensorFlow pour de meilleures performances sur M1.
    """
    # Activer Metal pour l'accélération GPU
    os.environ['TF_METAL_ENABLED'] = '1'
    
    # Limiter la mémoire GPU utilisée
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        # Permettre la croissance de la mémoire au lieu de préallouer tout
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    
    # Utiliser mixed_float16 pour améliorer les performances
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    
    # Optimiser le threading
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(8)  # 8 cœurs sur M1
    
    # Désactiver les opérations qui ne sont pas bien optimisées pour M1
    os.environ['TF_DISABLE_DEPTHWISE_CONV'] = '1'
    
    print("TensorFlow configuré pour Apple M1")
```

### 6.2 Optimisation de la Mémoire

```python
# src/utils/memory_utils.py
import gc
import tensorflow as tf

def optimize_memory():
    """
    Optimise l'utilisation de la mémoire.
    """
    # Forcer la collecte des déchets
    gc.collect()
    
    # Effacer la session TensorFlow
    tf.keras.backend.clear_session()

def batch_process_large_dataset(dataset, batch_size, process_func):
    """
    Traite un grand dataset par lots pour économiser la mémoire.
    """
    results = []
    
    for i in range(0, len(dataset), batch_size):
        # Traiter un lot
        batch = dataset[i:i+batch_size]
        batch_results = process_func(batch)
        results.extend(batch_results)
        
        # Optimiser la mémoire après chaque lot
        optimize_memory()
    
    return results
```

## 7. Script d'Exécution Principal

```python
# src/main.py
import os
from flask import Flask, request, jsonify, render_template, send_from_directory
import uuid
from werkzeug.utils import secure_filename

from src.config import configure_tensorflow_for_m1
from src.api.utils import ImageToAudioPipeline

# Configurer TensorFlow pour M1
configure_tensorflow_for_m1()

# Initialiser l'application Flask
app = Flask(__name__, 
            static_folder='../static',
            template_folder='../templates')

# Configurer les répertoires
UPLOAD_FOLDER = 'static/uploads'
AUDIO_FOLDER = 'static/audio'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(AUDIO_FOLDER, exist_ok=True)

# Initialiser le pipeline
pipeline = ImageToAudioPipeline(model_dir='data/models')

@app.route('/')
def index():
    """
    Page d'accueil.
    """
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Endpoint pour télécharger une image.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'Aucun fichier trouvé'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'Aucun fichier sélectionné'}), 400
    
    if file:
        # Générer un nom de fichier unique
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        
        # Sauvegarder le fichier
        file.save(file_path)
        
        # Traiter l'image
        try:
            result = pipeline.process_image(file_path)
            
            # Préparer la réponse
            response = {
                'success': True,
                'image_url': f"/static/uploads/{unique_filename}",
                'caption': result['caption'],
                'audio_url': f"/{result['audio_path']}",
                'processing_time': 0  # À implémenter si nécessaire
            }
            
            return jsonify(response)
        
        except Exception as e:
            return jsonify({'error': str(e)}), 500

@app.route('/static/audio/<filename>')
def serve_audio(filename):
    """
    Sert les fichiers audio.
    """
    return send_from_directory(AUDIO_FOLDER, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
```

## 8. Interface Utilisateur Simple

```html
<!-- templates/index.html -->
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Convertisseur Image-to-Audio</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.3/dist/css/bootstrap.min.css">
    <style>
        .container {
            max-width: 800px;
            margin-top: 50px;
        }
        .result-container {
            margin-top: 30px;
            display: none;
        }
        .image-preview {
            max-width: 100%;
            max-height: 400px;
            margin-bottom: 20px;
        }
        .loading {
            display: none;
            text-align: center;
            margin: 20px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="text-center mb-4">Convertisseur Image-to-Audio</h1>
        
        <div class="card">
            <div class="card-body">
                <h5 class="card-title">Télécharger une image</h5>
                <form id="upload-form" enctype="multipart/form-data">
                    <div class="mb-3">
                        <input class="form-control" type="file" id="image-input" accept="image/*">
                    </div>
                    <button type="submit" class="btn btn-primary">Convertir en Audio</button>
                </form>
                
                <div class="loading">
                    <div class="spinner-border text-primary" role="status">
                        <span class="visually-hidden">Chargement...</span>
                    </div>
                    <p>Traitement en cours, veuillez patienter...</p>
                </div>
            </div>
        </div>
        
        <div class="result-container card">
            <div class="card-body">
                <h5 class="card-title">Résultat</h5>
                
                <div class="text-center">
                    <img id="image-preview" class="image-preview" src="" alt="Image téléchargée">
                </div>
                
                <div class="mb-3">
                    <h6>Description générée:</h6>
                    <p id="caption-text" class="p-2 bg-light"></p>
                </div>
                
                <div class="mb-3">
                    <h6>Audio:</h6>
                    <audio id="audio-player" controls class="w-100"></audio>
                </div>
                
                <div class="d-flex justify-content-between">
                    <button id="download-audio" class="btn btn-success">Télécharger l'Audio</button>
                    <button id="new-conversion" class="btn btn-secondary">Nouvelle Conversion</button>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const form = document.getElementById('upload-form');
            const imageInput = document.getElementById('image-input');
            const loading = document.querySelector('.loading');
            const resultContainer = document.querySelector('.result-container');
            const imagePreview = document.getElementById('image-preview');
            const captionText = document.getElementById('caption-text');
            const audioPlayer = document.getElementById('audio-player');
            const downloadAudio = document.getElementById('download-audio');
            const newConversion = document.getElementById('new-conversion');
            
            form.addEventListener('submit', function(e) {
                e.preventDefault();
                
                const file = imageInput.files[0];
                if (!file) {
                    alert('Veuillez sélectionner une image');
                    return;
                }
                
                const formData = new FormData();
                formData.append('file', file);
                
                // Afficher le chargement
                loading.style.display = 'block';
                resultContainer.style.display = 'none';
                
                // Envoyer la requête
                fetch('/upload', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        // Afficher les résultats
                        imagePreview.src = data.image_url;
                        captionText.textContent = data.caption;
                        audioPlayer.src = data.audio_url;
                        
                        // Configurer le bouton de téléchargement
                        downloadAudio.onclick = function() {
                            const a = document.createElement('a');
                            a.href = data.audio_url;
                            a.download = 'audio_description.mp3';
                            document.body.appendChild(a);
                            a.click();
                            document.body.removeChild(a);
                        };
                        
                        // Afficher les résultats
                        resultContainer.style.display = 'block';
                    } else {
                        alert('Erreur: ' + data.error);
                    }
                })
                .catch(error => {
                    console.error('Erreur:', error);
                    alert('Une erreur est survenue lors du traitement de l\'image');
                })
                .finally(() => {
                    loading.style.display = 'none';
                });
            });
            
            newConversion.addEventListener('click', function() {
                // Réinitialiser le formulaire
                form.reset();
                resultContainer.style.display = 'none';
            });
        });
    </script>
</body>
</html>
```

## 9. Installation et Exécution

### 9.1 Requirements.txt

```
# requirements.txt
tensorflow-macos>=2.9.0
tensorflow-metal>=0.5.0
flask>=2.0.1
pillow>=8.3.1
numpy>=1.19.5
pandas>=1.3.3
matplotlib>=3.4.3
gtts>=2.2.3
pyttsx3>=2.90
requests>=2.26.0
werkzeug>=2.0.1
```

### 9.2 Script d'Installation pour MacBook M1

```bash
#!/bin/bash
# install.sh

# Créer un environnement virtuel
echo "Création de l'environnement virtuel..."
python3 -m venv venv

# Activer l'environnement
echo "Activation de l'environnement..."
source venv/bin/activate

# Mettre à jour pip
echo "Mise à jour de pip..."
pip install --upgrade pip

# Installer les dépendances pour M1
echo "Installation des dépendances pour M1..."
pip install -r requirements.txt

# Télécharger le dataset Flickr8k
echo "Téléchargement du dataset Flickr8k..."
python -c "from src.data.dataset import download_flickr8k; download_flickr8k()"

echo "Installation terminée avec succès!"
echo "Pour lancer l'application, exécutez: python src/main.py"
```

### 9.3 Script de Lancement

```bash
#!/bin/bash
# run.sh

# Activer l'environnement virtuel
source venv/bin/activate

# Lancer l'application Flask
python src/main.py
```

## Conclusion

Ce pipeline complet de prétraitement et d'entraînement est spécifiquement optimisé pour un MacBook Pro M1 avec 16GB de RAM. Il utilise:

1. **Dataset Flickr8k** pour un entraînement local efficace
2. **TensorFlow avec optimisations Metal** pour l'accélération matérielle
3. **Techniques d'économie de mémoire** (générateurs, traitement par lots)
4. **Architecture en deux étapes** (Image Captioning + TTS)
5. **Backend Flask** pour servir l'application

Les optimisations spécifiques pour M1 incluent:
- Utilisation de `mixed_float16` pour réduire l'empreinte mémoire
- Configuration du threading pour les 8 cœurs du M1
- Libération proactive de la mémoire
- Chargement des données par lots
- Utilisation de Metal pour l'accélération GPU

Cette implémentation permet un entraînement local complet sur un MacBook Pro M1 avec 16GB de RAM, tout en offrant des performances raisonnables pour un projet académique.
