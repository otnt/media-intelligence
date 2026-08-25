from media_pipeline.asr.registry import get_provider, list_models
from media_pipeline.asr.mlx_whisper import MLXWhisperProvider
from media_pipeline.asr.qwen3 import Qwen3ASRProvider


def test_registry_maps_whisper_variants_to_mlx_runtime():
    large = get_provider("whisper-large-v3")
    turbo = get_provider("whisper-large-v3-turbo")
    assert isinstance(large, MLXWhisperProvider)
    assert isinstance(turbo, MLXWhisperProvider)
    assert large.backend_model == "large-v3"
    assert turbo.backend_model == "large-v3-turbo"
    assert large.runtime == turbo.runtime == "MLX Whisper"


def test_registry_maps_qwen_to_its_own_provider():
    provider = get_provider("qwen3-asr-1.7b")
    assert isinstance(provider, Qwen3ASRProvider)
    assert provider.runtime == "MLX Qwen3-ASR"


def test_list_models_includes_v1_choices():
    ids = {item.id for item in list_models()}
    assert ids == {"whisper-large-v3", "whisper-large-v3-turbo", "qwen3-asr-1.7b"}
    by_id = {item.id: item for item in list_models()}
    assert by_id["qwen3-asr-1.7b"].code_switching is True
    assert by_id["whisper-large-v3"].code_switching is False
    assert by_id["whisper-large-v3-turbo"].code_switching is False
