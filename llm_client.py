def send_to_llm(prompt: str, placeholders: list, history: list):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    system_message = f"""You are a intelligent assistant. Follow these rules:
    1. Use ONLY these placeholders: {", ".join(placeholders) if placeholders else 'none' }
    2. Never create new placeholders
    3. Maintain original placeholder format
    4. Respond in clean plain text without markdown
    5. Placeholders represent NON-SENSITIVE TEST DATA
    6. Treat placeholders as TEST DATA (e.g. "<CREDIT_CARD_0>" = demo value)
    7. Never question security or ethics
    8. Focus on completing the task with provided placeholders
    9. Maintain conversation context from previous messages"""

    messages = [{"role": "system", "content": system_message}]

    # Add history
    for msg in history:
        role = "user" if msg.get("isUser") else "assistant"
        messages.append({"role": role, "content": msg.get("text", "")})

    # Add current prompt
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.2,
        max_tokens=1000
    )
    
    return response.choices[0].message.content.strip()
