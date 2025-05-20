"""
Script d'entraînement optimisé pour les grands datasets.
Utilise tf.data.Dataset au lieu de générateurs Python personnalisés.
"""
import os
import sys
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard

# Désactiver le GPU pour éviter les problèmes de mémoire
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Ajouter le répertoire parent au chemin pour pouvoir importer les modules du projet
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.caption_model import define_model

def load_descriptions(filename):
    """
    Charge et parse les descriptions des images.
    """
    file = open(filename, 'r')
    doc = file.read()
    file.close()
    
    mapping = {}
    for line in doc.split('\n'):
        if len(line) < 2:
            continue
        tokens = line.split()
        if len(tokens) < 2:
            continue
            
        image_id = tokens[0].split('.')[0]
        image_desc = ' '.join(tokens[1:]).lower()
        image_desc = 'startseq ' + image_desc + ' endseq'
        
        if image_id not in mapping:
            mapping[image_id] = []
        mapping[image_id].append(image_desc)
    
    print(f"Nombre de descriptions chargées: {len(mapping)}")
    return mapping

def load_set(filename):
    """
    Charge les identifiants des images pour un ensemble spécifique.
    """
    file = open(filename, 'r')
    doc = file.read()
    file.close()
    
    dataset = []
    for line in doc.split('\n'):
        if len(line) < 1:
            continue
        identifier = line.split('.')[0]
        dataset.append(identifier)
    
    print(f"Nombre d'identifiants dans {os.path.basename(filename)}: {len(dataset)}")
    return set(dataset)

def create_tokenizer(descriptions):
    """
    Crée un tokenizer à partir des descriptions.
    """
    all_desc = []
    for key in descriptions.keys():
        for desc in descriptions[key]:
            all_desc.append(desc)
    
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

def create_sequences(descriptions, features, tokenizer, max_length, vocab_size, limit=None):
    """
    Crée des séquences pour l'entraînement.
    """
    # Filtrer les descriptions pour ne garder que celles qui ont des caractéristiques
    valid_descriptions = {k: v for k, v in descriptions.items() if k in features}
    
    # Vérifier s'il y a des descriptions valides
    if not valid_descriptions:
        print("ERREUR: Aucune description valide trouvée pour l'entraînement.")
        return None, None, None
    
    print(f"Nombre d'images valides: {len(valid_descriptions)}")
    
    # Listes pour stocker les données
    X1, X2, y = [], [], []
    
    # Compteur pour limiter le nombre d'échantillons
    count = 0
    
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
                out_seq_vec = np.zeros(vocab_size)
                out_seq_vec[out_seq] = 1
                
                # Caractéristiques de l'image
                image_feature = features[image_id]
                
                # Ajouter aux listes
                X1.append(image_feature)
                X2.append(in_seq)
                y.append(out_seq_vec)
                
                # Incrémenter le compteur
                count += 1
                
                # Limiter le nombre d'échantillons si nécessaire
                if limit and count >= limit:
                    break
            
            # Limiter le nombre d'échantillons si nécessaire
            if limit and count >= limit:
                break
        
        # Limiter le nombre d'échantillons si nécessaire
        if limit and count >= limit:
            break
    
    # Convertir en tableaux numpy
    X1 = np.array(X1)
    X2 = np.array(X2)
    y = np.array(y)
    
    print(f"Données préparées: X1 shape={X1.shape}, X2 shape={X2.shape}, y shape={y.shape}")
    
    return X1, X2, y

def create_tf_dataset(X1, X2, y, batch_size=8, shuffle=True):
    """
    Crée un tf.data.Dataset à partir des données.
    """
    # Créer le dataset
    dataset = tf.data.Dataset.from_tensor_slices(((X1, X2), y))
    
    # Mélanger les données
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    
    # Créer des lots
    dataset = dataset.batch(batch_size)
    
    # Précharger les données
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return dataset

def setup_callbacks(checkpoint_path, log_dir):
    """
    Configure les callbacks pour l'entraînement.
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
        patience=10, # 10 epochs
        verbose=1,
        restore_best_weights=True
    )
    
    # Callback pour réduire le taux d'apprentissage
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=10, # 10 epochs
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

def main():
    """
    Fonction principale pour l'entraînement du modèle.
    """
    print("Début de l'entraînement...")
    
    # Définir les chemins
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
        print("Exécutez d'abord le script d'extraction des caractéristiques.")
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
    
    # Limiter le nombre d'échantillons pour éviter les problèmes de mémoire
    # Utilisez None pour traiter tous les échantillons, ou un nombre comme 50000 pour limiter
    sample_limit = 25000  # Limiter à 50000 échantillons
    
    # Créer les séquences
    print("Création des séquences d'entraînement...")
    train_X1, train_X2, train_y = create_sequences(train_descriptions, train_features, tokenizer, max_seq_length, vocab_size, limit=sample_limit)
    
    print("Création des séquences de validation...")
    val_X1, val_X2, val_y = create_sequences(val_descriptions, val_features, tokenizer, max_seq_length, vocab_size, limit=sample_limit//5)
    
    # Vérifier si les séquences ont été créées
    if train_X1 is None or val_X1 is None:
        print("ERREUR: Impossible de créer les séquences.")
        return
    
    # Créer les datasets TensorFlow
    batch_size = 32  # Vous pouvez ajuster cette valeur en fonction de votre mémoire
    train_dataset = create_tf_dataset(train_X1, train_X2, train_y, batch_size=batch_size)
    val_dataset = create_tf_dataset(val_X1, val_X2, val_y, batch_size=batch_size, shuffle=False)
    
    # Entraîner le modèle
    print("Début de l'entraînement...")
    history = model.fit(
        train_dataset,
        epochs=25,
        validation_data=val_dataset,
        callbacks=callbacks,
        verbose=1
    )
    
    # Sauvegarder le modèle final
    final_model_path = os.path.join(model_dir, 'final_model.h5')
    model.save(final_model_path)
    print(f"Modèle final sauvegardé dans {final_model_path}")
    
    print("Entraînement terminé avec succès!")

if __name__ == "__main__":
    main()
