"""
Routes pour l'API Flask.
"""
from flask import Blueprint, request, jsonify,url_for, render_template, send_from_directory
import os
import uuid
from werkzeug.utils import secure_filename

from src.api.utils import ImageToAudioPipeline

# Créer un blueprint pour les routes API
api_bp = Blueprint('api', __name__)

# Initialiser le pipeline
pipeline = None

def init_pipeline(model_dir):
    """
    Initialise le pipeline de traitement.
    
    Args:
        model_dir (str): Répertoire contenant les modèles
    """
    global pipeline
    pipeline = ImageToAudioPipeline(model_dir=model_dir)

@api_bp.route('/convert', methods=['POST'])
def convert():
    """
    Endpoint pour convertir une image en audio.
    
    Returns:
        JSON: Résultat de la conversion
    """
    if 'file' not in request.files:
        return jsonify({'error': 'Aucun fichier trouvé'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'Aucun fichier sélectionné'}), 400
    
    if file:
        # Générer un nom de fichier unique
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        file_path = os.path.join(api_bp.config['UPLOAD_FOLDER'], unique_filename)
        
        # Sauvegarder le fichier
        file.save(file_path)
        
        # Traiter l'image
        try:
            # Obtenir la langue depuis les paramètres (par défaut: français)
            lang = request.form.get('lang', 'fr')
            
            # Traiter l'image
            result = pipeline.process_image(file_path, lang=lang)
            audio_filename = os.path.basename(result[1]) 
            # Préparer la réponse
            response = {
                'success': True,
                'image_url': f"/static/uploads/{unique_filename}",
                'caption': result[0],
                'audio_url': url_for('serve_audio', filename=audio_filename),
                'processing_time': 0  # À implémenter si nécessaire
            }
            
            return jsonify(response)
        
        except Exception as e:
            return jsonify({'error': str(e)}), 500

@api_bp.route('/models', methods=['GET'])
def list_models():
    """
    Liste les modèles disponibles.
    
    Returns:
        JSON: Liste des modèles
    """
    model_dir = api_bp.config['MODEL_DIR']
    models = []
    
    # Vérifier si le modèle par défaut existe
    if os.path.exists(os.path.join(model_dir, 'best_model.h5')):
        models.append({
            'id': 'default',
            'name': 'CNN-LSTM Standard',
            'description': 'Modèle par défaut entraîné sur Flickr8k'
        })
    
    # Vérifier si le modèle final existe
    if os.path.exists(os.path.join(model_dir, 'final_model.h5')):
        models.append({
            'id': 'final',
            'name': 'Modèle final',
            'description': 'Dernière version du modèle entraîné'
        })
    
    return jsonify({'models': models})
