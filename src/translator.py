import os
from ollama import Client

OLLAMA_URL = os.getenv("OLLAMA_HOST", "localhost:11434")
client = Client(host=OLLAMA_URL)
MODEL_NAME = os.getenv("MODEL_NAME", "gemma3:270m")

# Translation context
TRANSLATION_CONTEXT = """You are a language translator. Your task is to translate non-English text into English.

CRITICAL RULES:
- Output ONLY the English translation
- Do NOT output the original text
- Do NOT explain or add any extra words
- Do NOT say "Translation:" or any prefix
- If the input is already in English, output it as-is
- Be natural and accurate

Examples:
INPUT: Bonjour, je m'appelle Bob
OUTPUT: Hello, my name is Bob

INPUT: Können Sie mir bitte helfen?
OUTPUT: Can you please help me?

INPUT: ¡Hola! ¿Cómo estás?
OUTPUT: Hello! How are you?

INPUT: 你好我是
OUTPUT: Hello, I am

INPUT: 你好Andrew！你今天过的怎么样？
OUTPUT: Hello Andrew! How are you doing today?

INPUT: Hello! How are you?
OUTPUT: Hello! How are you?
"""

# Classification context
CLASSIFICATION_CONTEXT = """You are a language classifier. Your task is to identify what language the input text is written in.

CRITICAL RULES:
- Output ONLY the language name in English (e.g., "English", "Chinese", "French", "German", "Spanish")
- Do NOT output the text itself
- Do NOT translate the text
- Do NOT explain or add any extra words
- For any form of Chinese (Simplified, Traditional, Cantonese), output only "Chinese"
- If unsure, output "Unknown"

Examples:
INPUT: Bonjour, je m'appelle Bob
OUTPUT: French

INPUT: Können Sie mir bitte helfen?
OUTPUT: German

INPUT: ¡Hola! ¿Cómo estás?
OUTPUT: Spanish

INPUT: Hello! How are you?
OUTPUT: English

INPUT: 你好Andrew！你今天过的怎么样？
OUTPUT: Chinese

INPUT: 你好我是
OUTPUT: Chinese
"""


def get_language(post: str) -> str:
    """Detect the language of the input text."""
    response = client.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": CLASSIFICATION_CONTEXT},
            {"role": "user", "content": f"INPUT: {post}"}
        ]
    )
    return response.message.content.strip()


def get_translation(post: str) -> str:
    """Translate the input text to English."""
    response = client.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": TRANSLATION_CONTEXT},
            {"role": "user", "content": f"INPUT: {post}"}
        ]
    )
    return response.message.content.strip()


def translate_content(content: str) -> tuple[bool, str]:
    """
    Robust translation function that handles errors gracefully.
    Returns (is_english, translated_content) tuple.
    """
    try:
        language = get_language(content)
        print(f"DEBUG: Detected language: '{language}'")
        
        # Check if LLM returned the actual text instead of language name
        if language == content or language in content or content in language:
            print(f"DEBUG: Language detection failed (returned text instead of language name), assuming non-English")
            is_english = False
        else:
            is_english = language.strip().lower() == "english"
        
        if is_english:
            return (True, content)
        else:
            # For any non-English language, attempt translation
            translation = get_translation(content)
            print(f"DEBUG: Translation result: '{translation}'")
            
            # Check if translation failed (LLM returned original text or empty)
            if translation == content or not translation.strip():
                print(f"DEBUG: Translation failed, returning original content as non-English")
                return (False, content)  # Return original but mark as non-English
            
            return (False, translation)
    except Exception as e:
        # Catch any other exceptions during the process and return original post
        print(f"Error processing post: {e}")
        return (True, content)  # Assume English and return original post as graceful fallback
