import os
from ollama import Client

OLLAMA_URL = os.getenv("OLLAMA_HOST", "localhost:11434")
client = Client(host=OLLAMA_URL)
MODEL_NAME = os.getenv("MODEL_NAME", "llama3.1:8b")

# Translation context
TRANSLATION_CONTEXT = """You are a language translator. Translate to English if it is not already in English. Follow the examples and do not output extra things.

Example:
INPUT: Bonjour, je m'appelle Bob
OUTPUT: Hello, my name is Bob

INPUT: Können Sie mir bitte helfen?
OUTPUT: Can you please help me?

INPUT: ¡Hola! ¿Cómo estás?
OUTPUT: Hello! How are you?

INPUT: Hello! How are you?
OUTPUT: Hello! How are you?
"""

# Classification context
CLASSIFICATION_CONTEXT = """You are a language classifier. Detect the language of the input text and reply only with the English name of that language. If you don't know the language, reply with "Unknown".
If it is a combination of multiple languages, reply with the most likely language with the most percentage in the text, ALWAYS output only one language instead of multiple ones.
For simplified Chinese, tradional Chinese, Cantonese, or other Chinese dialect, reply the word "Chinese".
For, English, don't reply with English-US or English_GB, but rather just "English".
Example:
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
"""


def get_language(post: str) -> str:
    """Detect the language of the input text."""
    response = client.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": CLASSIFICATION_CONTEXT},
            {"role": "user", "content": f"Detect the language of the following text:\n{post}"}
        ]
    )
    return response.message.content


def get_translation(post: str) -> str:
    """Translate the input text to English."""
    response = client.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": TRANSLATION_CONTEXT},
            {"role": "user", "content": f"Translate the following text to English. Only give translation and nothing else.\n{post}"}
        ]
    )
    return response.message.content


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
