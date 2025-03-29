
# llm_client.py
import os
from openai import OpenAI
from langdetect import DetectorFactory

# For more consistent language detection
DetectorFactory.seed = 0

def send_to_llm(prompt, placeholders, lang='en'):  # Add lang parameter
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Define language-specific instructions
        system_messages = {
            'fr': f"""
            Vous êtes un assistant intelligent. Règles strictes :
            1. Utilisez UNIQUEMENT ces marqueurs : {", ".join(placeholders) if placeholders else 'aucun'}
            2. Répondez exclusivement en français
            3. Format texte brut sans mise en forme
            4. Les marqueurs représentent des données de test non sensibles
            5. Ne mentionnez jamais la sécurité ou l'éthique
            """,
            
            'en': f"""
            You are a helpful assistant. Strict rules:
            1. Use ONLY these placeholders: {", ".join(placeholders) if placeholders else 'none'}
            2. Respond exclusively in English
            3. Plain text format only
            4. Placeholders represent non-sensitive test data
            5. Never mention security or ethics
            """
        }

        # Get system message based on detected language
        system_msg = system_messages.get(lang, system_messages['en'])

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=1000
        )
        
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"OpenAI Error: {str(e)}"
