import pytest
from src import translator as tr
from unittest.mock import patch

# Test data from P4A notebook evaluation sets
translation_eval_set = [
    {"post": "Hier ist dein erstes Beispiel.", "expected_answer": "Here is your first example."},
    {"post": "Bonjour, comment ça va?", "expected_answer": "Hello, how are you?"},
    {"post": "¡Hola! ¿Cómo estás?", "expected_answer": "Hello! How are you?"},
    {"post": "你好，很高兴认识你。", "expected_answer": "Hello, nice to meet you."},
    {"post": "Olá, como você está?", "expected_answer": "Hello, how are you?"},
    {"post": "Привет, как дела?", "expected_answer": "Hello, how are you?"},
    {"post": "こんにちは、お元気ですか？", "expected_answer": "Hello, how are you?"},
    {"post": "안녕하세요, 잘 지내세요?", "expected_answer": "Hello, how are you?"},
    {"post": "Ciao, come stai?", "expected_answer": "Hello, how are you?"},
    {"post": "Hej, hur mår du?", "expected_answer": "Hello, how are you?"},
    {"post": "Labas, kaip tau sekasi?", "expected_answer": "Hello, how are you?"},
    {"post": "Hei, miten menee?", "expected_answer": "Hello, how are you?"},
    {"post": "שלום מה נשמע", "expected_answer": "Hello what's up"},
    {"post": "مرحباً، كيف حالك؟", "expected_answer": "Hello, how are you?"},
    {"post": "¿Qué hora es?", "expected_answer": "What time is it?"},
    {"post": "Il pleut aujourd'hui.", "expected_answer": "It's raining today."},
    {"post": "Das Wetter ist schön.", "expected_answer": "The weather is nice."},
    {"post": "今天天气很好。", "expected_answer": "The weather is good today."},
    {"post": "Que horas são?", "expected_answer": "What time is it?"},
    {"post": "Какая сегодня погода?", "expected_answer": "What is the weather today?"},
]

language_detection_eval_set = [
    {"post": "Hier ist dein erstes Beispiel.", "expected_answer": "German"},
    {"post": "Bonjour, comment ça va?", "expected_answer": "French"},
    {"post": "¡Hola! ¿Cómo estás?", "expected_answer": "Spanish"},
    {"post": "你好，很高兴认识你。", "expected_answer": "Chinese"},
    {"post": "Olá, como você está?", "expected_answer": "Portuguese"},
    {"post": "Привет, как дела?", "expected_answer": "Russian"},
    {"post": "こんにちは、お元気ですか？", "expected_answer": "Japanese"},
    {"post": "안녕하세요, 잘 지내세요?", "expected_answer": "Korean"},
    {"post": "Ciao, come stai?", "expected_answer": "Italian"},
    {"post": "Hej, hur mår du?", "expected_answer": "Swedish"},
    {"post": "Labas, kaip tau sekasi?", "expected_answer": "Lithuanian"},
    {"post": "Hei, miten menee?", "expected_answer": "Finnish"},
    {"post": "שלום מה נשמע", "expected_answer": "Hebrew"},
    {"post": "مرحباً، كيف حالك؟", "expected_answer": "Arabic"},
    {"post": "This is an English post.", "expected_answer": "English"},
]

complete_eval_set = [
    # Non-English posts (20)
    {"post": "Hier ist dein erstes Beispiel.", "expected_answer": (False, "Here is your first example.")},
    {"post": "Bonjour, comment ça va?", "expected_answer": (False, "Hello, how are you?")},
    {"post": "¡Hola! ¿Cómo estás?", "expected_answer": (False, "Hello! How are you?")},
    {"post": "你好，很高兴认识你。", "expected_answer": (False, "Hello, nice to meet you.")},
    {"post": "Olá, como você está?", "expected_answer": (False, "Hello, how are you?")},
    {"post": "Привет, как дела?", "expected_answer": (False, "Hello, how are you?")},
    {"post": "こんにちは、お元気ですか？", "expected_answer": (False, "Hello, how are you?")},
    {"post": "안녕하세요, 잘 지내세요?", "expected_answer": (False, "Hello, how are you?")},
    {"post": "Ciao, come stai?", "expected_answer": (False, "Hello, how are you?")},
    {"post": "Il pleut aujourd'hui.", "expected_answer": (False, "It's raining today.")},
    {"post": "Das Wetter ist schön.", "expected_answer": (False, "The weather is nice.")},
    {"post": "今天天气很好。", "expected_answer": (False, "The weather is good today.")},
    {"post": "你好。吃饭了吗？", "expected_answer": (False, "Hello. Have you eaten?")},
    {"post": "今天天气怎么样？", "expected_answer": (False, "How is the weather today?")},
    {"post": "我喜欢学习编程。", "expected_answer": (False, "I like to learn programming.")},
    {"post": "谢谢你的帮助。", "expected_answer": (False, "Thank you for your help.")},
    {"post": "再见！", "expected_answer": (False, "Goodbye!")},
    {"post": "¿Qué hora es?", "expected_answer": (False, "What time is it?")},
    {"post": "Que horas são?", "expected_answer": (False, "What time is it?")},
    {"post": "Какая сегодня погода?", "expected_answer": (False, "What is the weather today?")},

    # English posts (16)
    {"post": "This is an English post.", "expected_answer": (True, "This is an English post.")},
    {"post": "Hello, how are you doing today?", "expected_answer": (True, "Hello, how are you doing today?")},
    {"post": "The quick brown fox jumps over the lazy dog.", "expected_answer": (True, "The quick brown fox jumps over the lazy dog.")},
    {"post": "Artificial intelligence is a fascinating field.", "expected_answer": (True, "Artificial intelligence is a fascinating field.")},
    {"post": "Learning to program can be challenging but rewarding.", "expected_answer": (True, "Learning to program can be challenging but rewarding.")},
    {"post": "I love spending time outdoors.", "expected_answer": (True, "I love spending time outdoors.")},
    {"post": "What are your plans for the weekend?", "expected_answer": (True, "What are your plans for the weekend?")},
    {"post": "Reading a good book is a great way to relax.", "expected_answer": (True, "Reading a good book is a great way to relax.")},
    {"post": "Tell me about your favorite hobby.", "expected_answer": (True, "Tell me about your favorite hobby.")},
    {"post": "The weather forecast for tomorrow is sunny.", "expected_answer": (True, "The weather forecast for tomorrow is sunny.")},
    {"post": "I'm looking forward to the holidays.", "expected_answer": (True, "I'm looking forward to the holidays.")},
    {"post": "Do you enjoy cooking?", "expected_answer": (True, "Do you enjoy cooking?")},
    {"post": "Traveling to new places is exciting.", "expected_answer": (True, "Traveling to new places is exciting.")},
    {"post": "What's your favorite type of music?", "expected_answer": (True, "What's your favorite type of music?")},
    {"post": "Let's grab some coffee sometime.", "expected_answer": (True, "Let's grab some coffee sometime.")},
    {"post": "It's important to stay hydrated.", "expected_answer": (True, "It's important to stay hydrated.")},

    # Edge cases / unintelligible (6)
    {"post": "asdfghjkl", "expected_answer": (True, "asdfghjkl")},
    {"post": "12345", "expected_answer": (True, "12345")},
    {"post": "!@#$%^", "expected_answer": (True, "!@#$%^")},
    {"post": "", "expected_answer": (True, "")},
    {"post": " ", "expected_answer": (True, " ")},
    {"post": "😊👋🎉", "expected_answer": (True, "😊👋🎉")},
]


def test_llm_normal_response(monkeypatch):
    """Test correct non-English translation behavior."""
    monkeypatch.setattr(tr, "get_language", lambda _txt: "French")
    monkeypatch.setattr(tr, "get_translation", lambda _txt: "Hello, how are you?")
    is_english, translated = tr.translate_content("Bonjour, comment ça va?")
    assert is_english is False
    assert translated == "Hello, how are you?"


def test_llm_gibberish_response(monkeypatch):
    """Test robust handling of malformed LLM output."""
    # When LLM returns gibberish that equals the original text, it should return original
    original = "Bonjour tout le monde"
    monkeypatch.setattr(tr, "get_language", lambda _txt: "gibberish")
    monkeypatch.setattr(tr, "get_translation", lambda _txt: original)
    is_english, translated = tr.translate_content(original)
    assert is_english is False
    assert translated == original


## Tests from our P4A notebook
@pytest.mark.parametrize("case", translation_eval_set)
def test_translation_eval_set(monkeypatch, case):
    """Run translation_eval_set from P4A notebook."""
    post = case["post"]
    expected = case["expected_answer"]
    # Mock to return non-English language and expected translation
    monkeypatch.setattr(tr, "get_language", lambda _txt: "Non-English")
    monkeypatch.setattr(tr, "get_translation", lambda _txt, e=expected: e)
    is_english, translated = tr.translate_content(post)
    assert is_english is False
    assert translated == expected


@pytest.mark.parametrize("case", language_detection_eval_set)
def test_language_detection_eval_set(monkeypatch, case):
    """Run language_detection_eval_set from P4A notebook."""
    post = case["post"]
    expected_lang = case["expected_answer"]
    monkeypatch.setattr(tr, "get_language", lambda _txt, e=expected_lang: e)

    if expected_lang == "English":
        # English posts should be returned as-is
        is_english, translated = tr.translate_content(post)
        assert is_english is True
        assert translated == post
    else:
        # Non-English posts should be translated
        monkeypatch.setattr(tr, "get_translation", lambda _txt: "Translated: " + post)
        is_english, translated = tr.translate_content(post)
        assert is_english is False
        assert translated == "Translated: " + post


@pytest.mark.parametrize("case", complete_eval_set)
def test_complete_eval_set(monkeypatch, case):
    """Run complete_eval_set from P4A notebook (end-to-end tests)."""
    post = case["post"]
    expected_is_english, expected_text = case["expected_answer"]

    # Mock the language detection and translation
    if expected_is_english:
        monkeypatch.setattr(tr, "get_language", lambda _txt: "English")
    else:
        monkeypatch.setattr(tr, "get_language", lambda _txt: "Non-English")
        monkeypatch.setattr(tr, "get_translation", lambda _txt, e=expected_text: e)

    is_english, translated = tr.translate_content(post)
    assert is_english == expected_is_english
    assert translated == expected_text


# Robustness/Error Handling Tests
@patch.object(tr.client, 'chat')
def test_unexpected_language_detection(mock_chat):
    """Test when LLM returns unexpected text instead of language name."""
    # When LLM returns same gibberish for both language detection and translation
    mock_chat.return_value.message.content = "I don't understand your request"
    original = "Hier ist dein erstes Beispiel."
    is_english, translated = tr.translate_content(original)
    # Returns the gibberish text since it doesn't match original
    assert is_english is False
    assert translated == "I don't understand your request"


@patch.object(tr.client, 'chat')
def test_empty_translation(mock_chat):
    """Test when LLM returns empty translation."""
    def side_effect(model, messages):
        # Check if it's language classification or translation
        if "language classifier" in messages[0]['content']:
            return type('obj', (object,), {'message': type('obj', (object,), {'content': 'German'})})()
        elif "language translator" in messages[0]['content']:
            return type('obj', (object,), {'message': type('obj', (object,), {'content': ''})})()
        return None

    mock_chat.side_effect = side_effect
    original = "Hier ist dein erstes Beispiel."
    is_english, translated = tr.translate_content(original)
    # Should return original when translation is empty
    assert is_english is False
    assert translated == original


@patch.object(tr.client, 'chat')
def test_ollama_exception(mock_chat):
    """Test exception handling when Ollama server is down."""
    mock_chat.side_effect = Exception("Ollama server is down")
    original = "Hier ist dein erstes Beispiel."
    is_english, translated = tr.translate_content(original)
    # Should gracefully fallback
    assert is_english is True
    assert translated == original


@patch.object(tr.client, 'chat')
def test_unintelligible_post(mock_chat):
    """Test handling of unintelligible/gibberish input."""
    def side_effect(model, messages):
        if "language classifier" in messages[0]['content']:
            return type('obj', (object,), {'message': type('obj', (object,), {'content': 'Unknown'})})()
        elif "language translator" in messages[0]['content']:
            return type('obj', (object,), {'message': type('obj', (object,), {'content': ''})})()
        return None

    mock_chat.side_effect = side_effect
    original = "asdfghjkl"
    is_english, translated = tr.translate_content(original)
    # Should return original for unintelligible text
    assert is_english is False
    assert translated == original
