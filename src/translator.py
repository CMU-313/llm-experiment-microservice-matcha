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
        # Handle unexpected language detection responses
        if language.strip().lower() not in ["english", "german", "french", "spanish", "chinese", "portuguese", "russian", "japanese", "korean", "italian", "swedish", "lithuanian", "finnish", "hebrew", "arabic"]:
            return (True, content)  # Graceful fallback for unexpected language

        is_english = language.strip().lower() == "english"
        if is_english:
            return (True, content)
        else:
            translation = get_translation(content)
            # Basic check for empty or unintelligible translation
            if not translation.strip():
                return (True, content)  # Graceful fallback for empty/unintelligible translation
            print(translation)
            return (False, translation)
    except Exception as e:
        # Catch any other exceptions during the process and return original post
        print(f"Error processing post: {e}")
        return (True, content)  # Assume English and return original post as graceful fallback
