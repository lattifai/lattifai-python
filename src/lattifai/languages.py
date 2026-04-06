"""Language registry for LattifAI.

Centralized language code mapping used across translation, transcription,
and caption processing. Codes follow BCP 47 / ISO 639-1 conventions.
"""

from typing import Dict, Optional

# (code, English name, native name)
_LANGUAGES = [
    # East Asian
    ("zh", "Chinese (Simplified)", "简体中文"),
    ("zh-TW", "Chinese (Traditional)", "繁體中文"),
    ("ja", "Japanese", "日本語"),
    ("ko", "Korean", "한국어"),
    # South / Southeast Asian
    ("hi", "Hindi", "हिन्दी"),
    ("bn", "Bengali", "বাংলা"),
    ("ta", "Tamil", "தமிழ்"),
    ("th", "Thai", "ไทย"),
    ("vi", "Vietnamese", "Tiếng Việt"),
    ("id", "Indonesian", "Bahasa Indonesia"),
    ("ms", "Malay", "Bahasa Melayu"),
    ("tl", "Filipino", "Filipino"),
    ("my", "Burmese", "မြန်မာ"),
    # Western European
    ("en", "English", "English"),
    ("es", "Spanish", "Español"),
    ("fr", "French", "Français"),
    ("de", "German", "Deutsch"),
    ("pt", "Portuguese", "Português"),
    ("pt-BR", "Portuguese (Brazilian)", "Português (Brasil)"),
    ("it", "Italian", "Italiano"),
    ("nl", "Dutch", "Nederlands"),
    ("ca", "Catalan", "Català"),
    # Northern European
    ("sv", "Swedish", "Svenska"),
    ("da", "Danish", "Dansk"),
    ("no", "Norwegian", "Norsk"),
    ("fi", "Finnish", "Suomi"),
    ("is", "Icelandic", "Íslenska"),
    # Eastern European
    ("ru", "Russian", "Русский"),
    ("uk", "Ukrainian", "Українська"),
    ("pl", "Polish", "Polski"),
    ("cs", "Czech", "Čeština"),
    ("sk", "Slovak", "Slovenčina"),
    ("ro", "Romanian", "Română"),
    ("hu", "Hungarian", "Magyar"),
    ("bg", "Bulgarian", "Български"),
    ("hr", "Croatian", "Hrvatski"),
    ("sr", "Serbian", "Српски"),
    # Southern European / Mediterranean
    ("el", "Greek", "Ελληνικά"),
    ("tr", "Turkish", "Türkçe"),
    # Middle Eastern
    ("ar", "Arabic", "العربية"),
    ("fa", "Persian", "فارسی"),
    ("he", "Hebrew", "עברית"),
    ("ur", "Urdu", "اردو"),
    # African
    ("sw", "Swahili", "Kiswahili"),
    ("am", "Amharic", "አማርኛ"),
]

# code -> English name
LANGUAGE_NAMES: Dict[str, str] = {code: name for code, name, _ in _LANGUAGES}

# code -> native name
LANGUAGE_NATIVE_NAMES: Dict[str, str] = {code: native for code, _, native in _LANGUAGES}

# All supported language codes
SUPPORTED_LANGUAGES = list(LANGUAGE_NAMES.keys())


def get_language_name(lang_code: str) -> str:
    """Get English language name from code. Returns the code itself if unknown."""
    return LANGUAGE_NAMES.get(lang_code, lang_code)


def get_native_name(lang_code: str) -> str:
    """Get native language name from code. Returns the code itself if unknown."""
    return LANGUAGE_NATIVE_NAMES.get(lang_code, lang_code)


def is_supported(lang_code: str) -> bool:
    """Check if a language code is supported."""
    return lang_code in LANGUAGE_NAMES


def find_language(query: str) -> Optional[str]:
    """Find language code by name (English or native), case-insensitive.

    Returns:
        Language code if found, None otherwise.
    """
    q = query.lower().strip()
    for code, name, native in _LANGUAGES:
        if q in (code.lower(), name.lower(), native.lower()):
            return code
    return None
