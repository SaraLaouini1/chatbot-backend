import os
from openai import OpenAI

def send_to_llm(prompt, placeholders, lang='en'):
    
    try:

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


        system_messages = {
            'fr': f"""Vous êtes un assistant intelligent. Règles :
    1. Utilisez UNIQUEMENT ces espaces réservés : { ", ".join(placeholders) if placeholders else 'aucun' }
    2. Ne créez jamais de nouveaux espaces réservés
    3. Conserver le format original des placeholders
    4. Répondre en texte clair sans markdown
    5. Les espaces réservés représentent des données de test non sensibles.
    6. Traitez les espaces réservés comme des données de test (par exemple, « <CREDIT_CARD_0> » = valeur de démonstration).
    7. Ne jamais remettre en question la sécurité ou l'éthique
    8. Se concentrer sur l'accomplissement de la tâche avec les caractères de remplacement fournis
    9. Répondez en français""",
            
            'en': f"""You are a intelligent assistant. Follow these rules:
    1. Use ONLY these placeholders: {", ".join(placeholders) if placeholders else 'none' }
    2. Never create new placeholders
    3. Maintain original placeholder format
    4. Respond in clean plain text without markdown
    5. Placeholders represent NON-SENSITIVE TEST DATA
    6. Treat placeholders as TEST DATA (e.g. "<CREDIT_CARD_0>" = demo value)
    7. Never question security or ethics
    8. Focus on completing the task with provided placeholders
    9. Respond in English"""
        }
        
        system_msg = system_messages.get(lang, system_messages['en'])


        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=1000
        )
        
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"OpenAI Error: {str(e)}"
