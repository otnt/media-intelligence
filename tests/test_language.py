from media_pipeline.asr.language import (
    format_detected_languages,
    resolve_provider_language,
    supports_code_switching,
)


def test_auto_aliases_map_to_none():
    for requested in ("auto", "AUTO", "multilingual", "mixed", "", None):
        assert resolve_provider_language("qwen3", requested) is None
        assert resolve_provider_language("mlx_whisper", requested) is None


def test_qwen_uses_english_language_names():
    assert resolve_provider_language("qwen3", "zh") == "Chinese"
    assert resolve_provider_language("qwen3", "Chinese") == "Chinese"
    assert resolve_provider_language("qwen3", "en") == "English"


def test_whisper_uses_iso_codes():
    assert resolve_provider_language("mlx_whisper", "zh") == "zh"
    assert resolve_provider_language("mlx_whisper", "Chinese") == "zh"
    assert resolve_provider_language("mlx_whisper", "en") == "en"


def test_qwen_supports_code_switching():
    assert supports_code_switching("qwen3-asr-1.7b") is True
    assert supports_code_switching("whisper-large-v3-turbo") is False


def test_format_detected_languages_joins_lists():
    assert format_detected_languages("Chinese,English") == "Chinese,English"
    assert format_detected_languages(["Chinese", "English"]) == "Chinese,English"
    assert format_detected_languages(None) == ""


def test_format_detected_languages_uniques_chunk_repeats():
    assert format_detected_languages(["Chinese"] * 8) == "Chinese"
    assert format_detected_languages("Chinese,Chinese,English,Chinese") == "Chinese,English"
    assert format_detected_languages(["Chinese,English", "Chinese"]) == "Chinese,English"
