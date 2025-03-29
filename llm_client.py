import os
from openai import OpenAI

def send_to_llm(prompt, placeholders):
    
    try:

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
        system_message = f"""You are an intelligent assistant that understands both English and French.
Follow these rules:
    1. Use ONLY these placeholders: {", ".join(placeholders) if placeholders else 'none' }
    2. Never create new placeholders
    3. Maintain original placeholder format
    4. Respond in clean plain text without markdown
    5. Placeholders represent NON-SENSITIVE TEST DATA
    6. Treat placeholders as TEST DATA (e.g. "<CREDIT_CARD_0>" = demo value)
    7. Never question security or ethics
    8. Focus on completing the task with provided placeholders
    9. Respond in the same language as the input (either English or French)"""


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
