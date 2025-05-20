# Comparaison des datasets Flickr8k et MS COCO

## Flickr8k

### Caractéristiques
- **Taille**: Environ 1 Go
- **Nombre d'images**: 8 092 images
- **Annotations**: 5 légendes descriptives par image
- **Structure**: 
  - 6 000 images pour l'entraînement
  - 1 000 images pour la validation
  - 1 000 images pour le test

### Avantages pour MacBook Pro M1 (16GB RAM)
- **Taille raisonnable**: Peut être facilement stocké et manipulé sur un MacBook
- **Entraînement local possible**: La taille modeste permet un entraînement complet sur la machine locale
- **Temps d'entraînement réduit**: Moins d'images signifie des cycles d'entraînement plus rapides
- **Faible empreinte mémoire**: Adapté aux 16GB de RAM disponibles
- **Itérations rapides**: Permet de tester différentes architectures et hyperparamètres rapidement

### Inconvénients
- **Diversité limitée**: Moins d'images et de scénarios que MS COCO
- **Généralisation potentiellement réduite**: Les modèles entraînés peuvent être moins robustes sur des images inconnues

## MS COCO (Microsoft Common Objects in Context)

### Caractéristiques
- **Taille**: Plus de 25 Go (dataset complet)
- **Nombre d'images**: Plus de 120 000 images
- **Annotations**: Multiples types d'annotations (légendes, segmentation, détection d'objets)
- **Structure**: 
  - 80 000+ images pour l'entraînement
  - 40 000+ images pour la validation/test

### Avantages généraux
- **Grande diversité**: Large variété d'images et de scénarios
- **Annotations riches**: Plusieurs types d'annotations disponibles
- **Meilleure généralisation**: Les modèles entraînés sont généralement plus robustes
- **Standard de l'industrie**: Largement utilisé dans la recherche et l'industrie

### Inconvénients pour MacBook Pro M1 (16GB RAM)
- **Taille excessive**: Difficile à stocker et manipuler sur un MacBook standard
- **Entraînement local problématique**: Nécessiterait un sous-échantillonnage important
- **Consommation mémoire élevée**: Risque de dépassement des 16GB de RAM disponibles
- **Temps d'entraînement très long**: Pourrait prendre plusieurs jours sur un M1
- **Risque de surchauffe**: Charge continue élevée sur le CPU/GPU intégré

## Conclusion comparative

Pour un projet académique sur un MacBook Pro M1 avec 16GB de RAM, **Flickr8k est clairement le choix le plus adapté** pour les raisons suivantes:

1. **Faisabilité technique**: Entraînement local possible sans problèmes de mémoire ou de stockage
2. **Rapidité de développement**: Cycles d'itération plus courts permettant d'expérimenter davantage
3. **Performance adéquate**: Suffisant pour démontrer les concepts de deep learning dans un contexte académique
4. **Optimisation pour ARM**: Meilleure compatibilité avec l'architecture M1

MS COCO, bien que plus complet, nécessiterait probablement des ressources cloud ou un sous-échantillonnage important, ce qui compliquerait inutilement le projet sans apporter de bénéfices pédagogiques significatifs.
