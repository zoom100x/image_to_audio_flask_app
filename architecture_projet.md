# Architecture du Projet Flask pour Conversion Image-to-Audio

## Structure du Projet

```
image_to_audio_app/
│
├── src/                          # Code source principal
│   ├── __init__.py               # Initialisation du package
│   ├── main.py                   # Point d'entrée de l'application Flask
│   ├── config.py                 # Configuration de l'application
│   │
│   ├── data/                     # Gestion des données
│   │   ├── __init__.py
│   │   ├── dataset.py            # Chargement et préparation du dataset Flickr8k
│   │   ├── preprocessing.py      # Prétraitement des images et textes
│   │   └── augmentation.py       # Augmentation de données (optionnel)
│   │
│   ├── models/                   # Modèles de deep learning
│   │   ├── __init__.py
│   │   ├── encoder.py            # Encodeur CNN pour les images
│   │   ├── decoder.py            # Décodeur LSTM/Transformer pour le texte
│   │   ├── caption_model.py      # Modèle complet d'image captioning
│   │   └── tts_model.py          # Modèle ou API de text-to-speech
│   │
│   ├── training/                 # Logique d'entraînement
│   │   ├── __init__.py
│   │   ├── trainer.py            # Classe d'entraînement du modèle
│   │   ├── callbacks.py          # Callbacks personnalisés
│   │   └── metrics.py            # Métriques d'évaluation
│   │
│   ├── api/                      # API REST Flask
│   │   ├── __init__.py
│   │   ├── routes.py             # Définition des routes API
│   │   ├── utils.py              # Utilitaires pour l'API
│   │   └── validation.py         # Validation des entrées
│   │
│   └── utils/                    # Utilitaires généraux
│       ├── __init__.py
│       ├── file_utils.py         # Gestion des fichiers
│       ├── visualization.py      # Visualisation des résultats
│       └── audio_utils.py        # Traitement audio
│
├── static/                       # Fichiers statiques pour l'interface web
│   ├── css/
│   ├── js/
│   └── images/
│
├── templates/                    # Templates HTML pour l'interface web
│   ├── index.html
│   ├── upload.html
│   └── result.html
│
├── tests/                        # Tests unitaires et d'intégration
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_models.py
│   └── test_api.py
│
├── data/                         # Dossier pour stocker les données
│   ├── flickr8k/                 # Dataset Flickr8k
│   ├── processed/                # Données prétraitées
│   └── models/                   # Modèles entraînés
│
├── notebooks/                    # Notebooks Jupyter pour l'exploration et le prototypage
│   ├── data_exploration.ipynb
│   ├── model_prototyping.ipynb
│   └── evaluation.ipynb
│
├── requirements.txt              # Dépendances du projet
├── setup.py                      # Script d'installation
├── README.md                     # Documentation principale
└── .gitignore                    # Fichiers à ignorer pour Git
```

## Flux de Données et Traitement

1. **Acquisition et Prétraitement des Données**
   - Téléchargement et extraction du dataset Flickr8k
   - Prétraitement des images (redimensionnement, normalisation)
   - Prétraitement des légendes (tokenization, padding)
   - Division en ensembles d'entraînement, validation et test
   - Création de générateurs de données optimisés pour la mémoire

2. **Entraînement du Modèle**
   - Initialisation du modèle d'image captioning (CNN+LSTM/Transformer)
   - Configuration des hyperparamètres adaptés au M1
   - Entraînement avec sauvegarde des meilleurs checkpoints
   - Évaluation sur l'ensemble de validation
   - Optimisations spécifiques pour Apple Silicon (batch size, quantization)

3. **Conversion Text-to-Speech**
   - Intégration d'une API TTS (gTTS, Amazon Polly) ou modèle local
   - Conversion des légendes générées en fichiers audio
   - Optimisation de la qualité et du naturel de la parole

4. **API Flask**
   - Endpoint pour télécharger une image
   - Traitement de l'image par le modèle entraîné
   - Génération de la légende textuelle
   - Conversion de la légende en audio
   - Retour des résultats (image, texte, audio) à l'utilisateur

## Diagramme de Séquence

```
Client                  API Flask               Modèle Caption          TTS
  |                        |                         |                    |
  | Upload Image           |                         |                    |
  |----------------------->|                         |                    |
  |                        | Prétraitement Image     |                    |
  |                        |------------------------>|                    |
  |                        |                         |                    |
  |                        |                         | Génération Légende |
  |                        |                         |------------------->|
  |                        |                         |<-------------------|
  |                        |<------------------------|                    |
  |                        |                         |                    |
  |                        | Légende                 |                    |
  |                        |------------------------------------------>  |
  |                        |                         |                    |
  |                        |                         |                    | Conversion Audio
  |                        |                         |                    |--------------->
  |                        |<------------------------------------------- |
  |                        |                         |                    |
  | Réponse (Image, Texte, Audio)                    |                    |
  |<-----------------------|                         |                    |
  |                        |                         |                    |
```

## Optimisations pour MacBook Pro M1

1. **Utilisation de TensorFlow-Metal**
   - Activation de l'accélération GPU via Metal
   - Configuration optimale pour l'architecture ARM

2. **Gestion de la Mémoire**
   - Chargement des données par lots (batch loading)
   - Libération proactive de la mémoire
   - Utilisation de générateurs au lieu de chargement complet

3. **Quantification des Modèles**
   - Réduction de la précision des poids (float16)
   - Optimisation pour les Neural Engine du M1

4. **Parallélisation Efficace**
   - Utilisation optimale des 8 cœurs du M1
   - Configuration du threading pour éviter la surcharge

## API REST

### Endpoints Principaux

1. **`POST /api/convert`**
   - **Description**: Convertit une image en audio
   - **Paramètres**: Image (multipart/form-data)
   - **Réponse**: 
     ```json
     {
       "success": true,
       "caption": "un chat orange assis sur un rebord de fenêtre",
       "audio_url": "/static/results/audio_12345.mp3",
       "processing_time": 1.23
     }
     ```

2. **`GET /api/models`**
   - **Description**: Liste les modèles disponibles
   - **Réponse**: 
     ```json
     {
       "models": [
         {
           "id": "default",
           "name": "CNN-LSTM Standard",
           "description": "Modèle par défaut entraîné sur Flickr8k"
         },
         {
           "id": "transformer",
           "name": "Vision Transformer",
           "description": "Modèle expérimental basé sur transformers"
         }
       ]
     }
     ```

3. **`POST /api/train`**
   - **Description**: Lance un entraînement personnalisé (admin)
   - **Paramètres**: Configuration d'entraînement (JSON)
   - **Réponse**: 
     ```json
     {
       "job_id": "train_12345",
       "status": "started",
       "estimated_time": "45 minutes"
     }
     ```

## Interface Utilisateur

L'interface web permettra:
1. Téléchargement d'images via glisser-déposer
2. Visualisation de l'image téléchargée
3. Affichage de la légende générée
4. Lecture du fichier audio généré
5. Historique des conversions récentes
6. Options de téléchargement des résultats
