"""
Point d'entrée principal de l'application Flask.
"""
import os
from flask import Flask, request, jsonify, render_template, send_from_directory
import uuid
from werkzeug.utils import secure_filename

from src.config import configure_tensorflow_for_m1, get_config
from src.api.routes import api_bp, init_pipeline

# Configurer TensorFlow pour M1
configure_tensorflow_for_m1()

def create_app(config_name='default'):
    """
    Crée et configure l'application Flask.
    
    Args:
        config_name (str): Nom de la configuration à utiliser
        
    Returns:
        Flask: Application Flask configurée
    """
    # Récupérer le chemin absolu vers la racine du projet
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # Initialiser Flask avec des chemins absolus
    app = Flask(__name__,
                static_folder=os.path.join(BASE_DIR, 'static'),
                template_folder=os.path.join(BASE_DIR, 'templates'))
    
    
    # Charger la configuration
    app.config.from_object(get_config(config_name))
    
    # Configurer les répertoires
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['AUDIO_FOLDER'], exist_ok=True)
    
    # Initialiser le pipeline
    init_pipeline(app.config['MODEL_DIR'])
    
    # Configurer le blueprint API
    api_bp.config = app.config
    app.register_blueprint(api_bp, url_prefix='/api')
    
    @app.route('/')
    def index():
        """
        Page d'accueil.
        """
        return render_template('index.html')
    
    @app.route('/static/audio/<filename>')
    def serve_audio(filename):
        """
        Sert les fichiers audio.
        """
        return send_from_directory(os.path.join(BASE_DIR, 'static', 'audio'), filename)
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5030, debug=True)
