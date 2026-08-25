from media_pipeline.asr.qwen3 import _stamps_to_words
from media_pipeline.models import WordSpan


def test_stamps_to_words_reads_mlx_segment_dicts():
    words = _stamps_to_words(
        [
            {"text": "姜", "start": 0.16, "end": 0.32},
            {"text": "Dora", "start": 0.32, "end": 0.64},
        ]
    )
    assert words == [
        WordSpan(start=0.16, end=0.32, text="姜"),
        WordSpan(start=0.32, end=0.64, text="Dora"),
    ]


def test_stamps_to_words_reads_qwen_aligner_items():
    class Item:
        def __init__(self, text, start_time, end_time):
            self.text = text
            self.start_time = start_time
            self.end_time = end_time

    class Result:
        def __init__(self, items):
            self.items = items

    words = _stamps_to_words(Result([Item("hello", 1.0, 1.4)]))
    assert words == [WordSpan(start=1.0, end=1.4, text="hello")]
