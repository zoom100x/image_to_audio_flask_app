"""
Configuration de l'application et optimisations pour M1.
"""
import tensorflow as tf
import os

def configure_tensorflow_for_m1():
    """
    Configure TensorFlow pour de meilleures performances sur M1.
    
    Cette fonction active les optimisations spécifiques pour l'architecture
    Apple Silicon M1, notamment l'accélération Metal et la gestion de mémoire.
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

# Configuration par défaut de l'application
class Config:
    """Configuration de base pour l'application Flask."""
    DEBUG = False
    TESTING = False
    SECRET_KEY = 'dev-key-change-in-production'
    UPLOAD_FOLDER = 'static/uploads'
    AUDIO_FOLDER = 'static/audio'
    MODEL_DIR = 'data/models'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB max upload

class DevelopmentConfig(Config):
    """Configuration pour le développement."""
    DEBUG = True

class ProductionConfig(Config):
    """Configuration pour la production."""
    SECRET_KEY = os.environ.get('SECRET_KEY', 'production-key')

# Dictionnaire des configurations
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}

def get_config(config_name='default'):
    """
    Récupère la configuration spécifiée.
    
    Args:
        config_name (str): Nom de la configuration ('development', 'production', 'default')
        
    Returns:
        Config: Objet de configuration
    """
    return config.get(config_name, config['default'])
