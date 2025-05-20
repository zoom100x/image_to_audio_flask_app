"""
Utilitaires pour la conversion de texte en audio.
"""
import os
import gtts
import pyttsx3

def text_to_speech_with_fallback(text, output_path, lang='fr'):
    """
    Convertit du texte en audio avec gTTS, avec pyttsx3 comme solution de secours.
    
    Args:
        text (str): Texte à convertir
        output_path (str): Chemin du fichier audio de sortie
        lang (str): Code de langue pour la synthèse vocale
        
    Returns:
        str: Chemin du fichier audio généré
    """
    # Créer le répertoire de sortie si nécessaire
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Essayer d'abord avec gTTS (nécessite une connexion Internet)
    try:
        tts = gtts.gTTS(text=text, lang=lang, slow=False)
        tts.save(output_path)
        print(f"Audio généré avec gTTS: {output_path}")
        return output_path
    except Exception as e:
        print(f"Erreur avec gTTS: {str(e)}")
        print("Utilisation de pyttsx3 comme solution de secours...")
        
        # Solution de secours avec pyttsx3 (hors ligne)
        try:
            engine = pyttsx3.init()
            
            # Configurer la voix
            voices = engine.getProperty('voices')
            if lang.startswith('fr'):
                # Chercher une voix française
                for voice in voices:
                    if 'french' in voice.name.lower() or 'fr' in voice.id.lower():
                        engine.setProperty('voice', voice.id)
                        break
            elif lang.startswith('en'):
                # Chercher une voix anglaise
                for voice in voices:
                    if 'english' in voice.name.lower() or 'en' in voice.id.lower():
                        engine.setProperty('voice', voice.id)
                        break
            
            # Configurer la vitesse
            engine.setProperty('rate', 150)
            
            # Générer l'audio
            engine.save_to_file(text, output_path)
            engine.runAndWait()
            
            print(f"Audio généré avec pyttsx3: {output_path}")
            return output_path
        except Exception as e:
            print(f"Erreur avec pyttsx3: {str(e)}")
            print("Impossible de générer l'audio.")
            return None
