# ASR + speaker benchmark

Compares ASR systems on transcript accuracy and speaker tags. Data lives outside the repo in `~/AIContent/bench/` (audio is large and partly third-party).

## Samples

| id | kind | language | length | speakers | ground truth |
|----|------|----------|--------|----------|--------------|
| `syn-zh` | Kokoro TTS | Chinese | ~3 min | 3 | exact text, speaker, start/end per utterance |
| `syn-en` | Kokoro TTS | English | ~3 min | 3 | exact |
| `syn-mix` | Kokoro TTS | Chinese–English code-switching | ~3 min | 3 | exact |
| `real-en-lex503` | Lex Fridman Podcast #503, first 15 min | English | 15 min | 4 | human transcript, per-turn speaker + start time |
| `real-zh-bmb232` | 不明白播客 EP-232, first 15 min | Chinese | 15 min | 2 | human-edited 文字版, per-turn speaker + start time |

Synthetic dialogues (`make_synthetic.py`) use different Kokoro voices per speaker, 0.15–0.7 s gaps (longer at speaker changes), short back-channels ("嗯", "Sure."), ±5% speaking rate jitter and −45 dBFS room tone.
Real samples (`fetch_real.py`) take audio from the YouTube video whose clock the transcript timestamps use; turn end = next turn's start.

Caveats:

- Real transcripts are cleaned or edited, not strictly verbatim (fillers dropped, light rewording), so real-sample ER has a floor above zero for every system.
- Numbers are not normalised ("四千五" vs "4500" counts as errors).
- Synthetic speech is cleaner than real speech and never overlaps.

## Running

```bash
# 1. synthetic audio (separate venv with kokoro, misaki[zh,en], soundfile)
VIRTUAL_ENV=~/AIContent/bench/.venv-tts ~/AIContent/bench/.venv-tts/bin/python bench/make_synthetic.py ~/AIContent/bench/samples
# 2. real podcasts
uv run python bench/fetch_real.py ~/AIContent/bench/samples
# 3. systems (results are cached per run; delete a file to redo it)
uv run python bench/run.py ~/AIContent/bench --systems agy-argon,fun-asr,qwen3-local --runs 2
# 4. scores
uv run --with jiwer --with scipy python bench/score.py ~/AIContent/bench
```

Systems hear audio only; no title/description context is passed.

## Metrics

- **ER**: token error rate after NFKC, lower-casing and punctuation removal. A token is one CJK character or one Latin word, so this is CER for Chinese, WER for English and MER for mixed speech.
- **cpER**: speaker-attributed ER. Each speaker's words are concatenated, hypothesis speakers are matched to reference speakers by the Hungarian algorithm, and errors are summed. A correct word under the wrong speaker counts as an error.
- **spk**: speakers found vs reference.
- **attrib err**: share of hypothesis speech time assigned to the wrong speaker under the best time-overlap mapping (10 ms frames).
- **DER**: missed + false-alarm + confusion over reference speech, 0.25 s collar. Synthetic samples only, since real transcripts have no turn ends.
- **chg**: for each reference speaker change, distance to the nearest hypothesis change after speaker mapping; median seconds and share within 1 s.

The scorer was checked against oracle hypotheses (all zeros) and half-swapped speakers (ER 0, speaker metrics up).
