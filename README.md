# Projet de Conversion Image vers Audio avec Deep Learning

Ce projet implémente une application de conversion d'images en audio en utilisant des techniques de deep learning. L'application analyse une image, génère une description textuelle de son contenu, puis convertit cette description en fichier audio.

## Architecture du Projet

Le projet est structuré comme suit :

```
image_to_audio_project/
├── src/
│   ├── api/                # API Flask pour l'interface web
│   ├── data/               # Gestion des données et prétraitement
│   ├── models/             # Modèles de deep learning
│   ├── training/           # Scripts d'entraînement
│   ├── utils/              # Utilitaires divers
│   ├── config.py           # Configuration globale
│   └── main.py             # Point d'entrée principal
├── data/                   # Données (dataset, caractéristiques, modèles)
│   ├── minimal/            # Dataset minimal synthétique
│   └── models/             # Modèles entraînés
├── templates/              # Templates HTML pour l'interface web
├── output/                 # Fichiers de sortie (audio générés)
├── requirements.txt        # Dépendances Python
├── install.sh              # Script d'installation
└── run.sh                  # Script d'exécution
```

## Optimisations pour MacBook Pro M1

Ce projet est spécifiquement optimisé pour fonctionner sur les MacBook Pro avec puce M1 :

- Utilisation de TensorFlow-Metal pour l'accélération GPU
- Configuration LSTM compatible avec l'architecture Apple Silicon
- Gestion efficace de la mémoire pour les 16GB de RAM
- Dataset Flickr8k (1GB) au lieu de MS COCO (25GB+)

## Installation

1. Clonez ce dépôt :
```bash
git clone <url-du-repo>
cd image_to_audio_app
```

2. Exécutez le script d'installation :
```bash
bash install.sh
```

Ce script créera un environnement virtuel Python et installera toutes les dépendances nécessaires.

## Utilisation

### Téléchargez le dataset Flickr8k :

Images : https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip
Textes : https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip

### Extrayez les fichiers dans les répertoires appropriés :

```
data/flickr8k/Flickr8k_Dataset/
data/flickr8k/Flickr8k_text/
```

### Création d'un Dataset Minimal

Pour créer un dataset minimal synthétique (recommandé pour les tests) :

```bash
python src/utils/create_synthetic_dataset.py
```
### pre-traitement du dataset et extraction des features :

Pretraitement  :

```bash
python src/data/preprocessing_fixed.py
```
Extraction des features  :

```bash
python src/data/extract_features.py
```

### Entraînement du Modèle

Pour entraîner le modèle avec le dataset minimal :

```bash
python -m src.training.train_tf_dataset
```

### Inférence

Pour générer une description audio à partir d'une image :

```bash
python src/utils/inference.py chemin/vers/votre/image.jpg
```

### Interface Web

Pour lancer l'interface web Flask :

```bash
bash run.sh
```

Puis accédez à `http://localhost:5000` dans votre navigateur.

## Pipeline de Traitement

1. **Extraction des caractéristiques** : Une image est traitée par un réseau de neurones convolutif (InceptionV3) pour extraire ses caractéristiques visuelles.

2. **Génération de légende** : Un modèle séquentiel (LSTM) utilise ces caractéristiques pour générer une description textuelle de l'image.

3. **Synthèse vocale** : La description textuelle est convertie en audio à l'aide de technologies de synthèse vocale (gTTS ou pyttsx3).

## Dépendances Principales

- TensorFlow-Metal >= 2.9.0
- Flask >= 2.0.1
- Pillow >= 8.3.1
- NumPy >= 1.19.5
- gTTS >= 2.2.3
- pyttsx3 >= 2.90

## Résolution des Problèmes Courants

### Erreur cuDNN avec LSTM

Si vous rencontrez une erreur liée à cuDNN et au masquage LSTM, assurez-vous d'utiliser la version optimisée du modèle (`caption_model_fixed_cudnn.py`).

### Problèmes de Dataset

En cas de problèmes avec le dataset Flickr8k, utilisez le script `create_synthetic_dataset.py` pour générer un dataset minimal fonctionnel.

### Erreurs de Mémoire

Si vous rencontrez des erreurs de mémoire, réduisez la taille du batch dans `train_tf_dataset.py` (par exemple, passez de 32 à 16).

## Licence

Ce projet est sous licence MIT. Voir le fichier LICENSE pour plus de détails.
