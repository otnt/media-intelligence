from __future__ import annotations

AUTO_ALIASES = {"", "auto", "multilingual", "multi", "any", "mix", "mixed"}

QWEN_LANGUAGE_NAMES = {
    "zh": "Chinese",
    "cn": "Chinese",
    "chinese": "Chinese",
    "en": "English",
    "eng": "English",
    "english": "English",
    "yue": "Cantonese",
    "cantonese": "Cantonese",
    "ja": "Japanese",
    "jp": "Japanese",
    "japanese": "Japanese",
    "ko": "Korean",
    "kr": "Korean",
    "korean": "Korean",
    "ar": "Arabic",
    "arabic": "Arabic",
    "de": "German",
    "german": "German",
    "fr": "French",
    "french": "French",
    "es": "Spanish",
    "spanish": "Spanish",
    "pt": "Portuguese",
    "portuguese": "Portuguese",
    "ru": "Russian",
    "russian": "Russian",
    "vi": "Vietnamese",
    "vietnamese": "Vietnamese",
    "th": "Thai",
    "thai": "Thai",
    "id": "Indonesian",
    "indonesian": "Indonesian",
    "it": "Italian",
    "italian": "Italian",
    "tr": "Turkish",
    "turkish": "Turkish",
    "hi": "Hindi",
    "hindi": "Hindi",
    "ms": "Malay",
    "malay": "Malay",
    "nl": "Dutch",
    "dutch": "Dutch",
    "sv": "Swedish",
    "swedish": "Swedish",
    "da": "Danish",
    "danish": "Danish",
    "fi": "Finnish",
    "finnish": "Finnish",
    "pl": "Polish",
    "polish": "Polish",
    "cs": "Czech",
    "czech": "Czech",
    "fil": "Filipino",
    "filipino": "Filipino",
    "fa": "Persian",
    "persian": "Persian",
    "el": "Greek",
    "greek": "Greek",
    "hu": "Hungarian",
    "hungarian": "Hungarian",
    "mk": "Macedonian",
    "macedonian": "Macedonian",
    "ro": "Romanian",
    "romanian": "Romanian",
}

WHISPER_LANGUAGE_CODES = {
    "zh": "zh",
    "cn": "zh",
    "chinese": "zh",
    "en": "en",
    "eng": "en",
    "english": "en",
    "yue": "yue",
    "cantonese": "yue",
    "ja": "ja",
    "jp": "ja",
    "japanese": "ja",
    "ko": "ko",
    "kr": "ko",
    "korean": "ko",
}

CODE_SWITCHING_MODELS = {"qwen3-asr-1.7b"}


def is_auto_language(requested: str | None) -> bool:
    return (requested or "auto").strip().lower() in AUTO_ALIASES


def resolve_provider_language(provider: str, requested: str | None) -> str | None:
    """Map a user language setting onto a provider-specific value.

    None means automatic detection. For Qwen3-ASR that also enables
    mixed-language / code-switched transcription.
    """
    if is_auto_language(requested):
        return None
    raw = (requested or "").strip()
    key = raw.lower()
    if provider == "qwen3":
        if raw in QWEN_LANGUAGE_NAMES.values():
            return raw
        return QWEN_LANGUAGE_NAMES.get(key, raw)
    if provider == "mlx_whisper":
        return WHISPER_LANGUAGE_CODES.get(key, key)
    return raw or None


def supports_code_switching(model_id: str) -> bool:
    return model_id in CODE_SWITCHING_MODELS


def format_detected_languages(value: object | None) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        parts = [str(item).strip() for item in value if str(item).strip()]
        return ",".join(parts)
    return str(value).strip()
