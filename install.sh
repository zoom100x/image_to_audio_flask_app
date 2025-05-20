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

