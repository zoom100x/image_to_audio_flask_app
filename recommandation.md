# Recommandation de Dataset et Framework

## Dataset recommandé: Flickr8k

Après analyse comparative détaillée, **Flickr8k** est clairement le choix optimal pour ce projet sur MacBook Pro M1 avec 16GB de RAM pour les raisons suivantes:

- **Taille gérable**: ~1 Go contre plus de 25 Go pour MS COCO
- **Entraînement local efficace**: 8 092 images (vs 120 000+ pour MS COCO)
- **Compatibilité avec les ressources disponibles**: Adapté aux 16GB de RAM
- **Temps d'entraînement raisonnable**: Cycles d'itération rapides pour le développement
- **Suffisant pour les objectifs académiques**: Permet de démontrer les concepts de deep learning

## Frameworks recommandés

### Backend: Flask

Flask est parfaitement adapté à ce projet pour plusieurs raisons:
- **Léger et flexible**: Idéal pour les applications de démonstration
- **Compatible ARM/M1**: Fonctionne nativement sur l'architecture Apple Silicon
- **Intégration simple avec les modèles ML**: API REST facile à implémenter
- **Déploiement simple**: Peut être déployé localement ou sur des services cloud
- **Documentation abondante**: Nombreuses ressources pour l'intégration avec TensorFlow/PyTorch

### Deep Learning: TensorFlow avec Keras

Pour l'architecture M1, nous recommandons:
- **TensorFlow 2.x avec Keras**: Optimisé pour Apple Silicon via le package `tensorflow-macos`
- **Installation via Miniforge**: Meilleure compatibilité avec l'architecture ARM

### Alternatives viables
- **PyTorch**: Également compatible M1 via Miniforge, offre plus de flexibilité mais avec une courbe d'apprentissage légèrement plus élevée
- **Hugging Face Transformers**: Pour utiliser des modèles pré-entraînés (recommandé pour accélérer le développement)

## Architecture recommandée pour la conversion Image-to-Audio

Pour un projet académique sur M1, l'approche en deux étapes est recommandée:
1. **Image Captioning**: CNN+LSTM/Transformer pour générer des descriptions textuelles
2. **Text-to-Speech**: Conversion des descriptions en audio via API (gTTS) ou modèle local

Cette approche offre le meilleur équilibre entre faisabilité technique, performance et valeur pédagogique.
