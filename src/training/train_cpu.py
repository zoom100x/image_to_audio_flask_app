# Désactiver le GPU avant d'importer TensorFlow
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Importer le reste du script train_final.py
exec(open("src/training/train_final.py").read())
