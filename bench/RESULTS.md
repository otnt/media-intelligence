# Benchmark results — 2026-11-10

Five samples (see README), systems hear audio only. `agy-argon` = Gemini 4 Argon via agy, 3-min windows + speaker roster, rescale off; `agy-argon-rescale` = same with rescale on (now the default). Fun-ASR = DashScope international. Qwen3 local has no diarization.

| sample | system | runs | ER | cpER | spk | attrib err | DER | chg med (s) | chg ≤1s | time (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| real-en-lex503 | agy-argon | 2 | 21.8% ±3.4 | 47.4% ±1.1 | 5/5 (ref 4) | 21.3% | — | 6.53 | 4.2% | 110.50 |
| real-en-lex503 | agy-argon-rescale | 1 | 15.2% | 40.7% | 5 (ref 4) | 22.7% | — | 5.58 | 25.0% | 142.50 |
| real-en-lex503 | fun-asr | 2 | 10.2% ±0.1 | 10.6% ±0.1 | 4/4 (ref 4) | 0.3% | — | 0.50 | 100.0% | 25.45 |
| real-en-lex503 | qwen3-local | 1 | 9.1% | — | — (ref 4) | — | — | — | — | 643.10 |
| real-zh-bmb232 | agy-argon | 2 | 35.1% ±0.1 | 56.9% ±5.9 | 20/14 (ref 2) | 22.3% | — | 6.09 | 20.0% | 142.65 |
| real-zh-bmb232 | agy-argon-rescale | 1 | 22.8% | 22.9% | 2 (ref 2) | 2.1% | — | 1.35 | 40.0% | 120.40 |
| real-zh-bmb232 | fun-asr | 2 | 22.3% ±0.0 | 22.2% ±0.0 | 2/2 (ref 2) | 0.6% | — | 0.54 | 86.7% | 34.20 |
| real-zh-bmb232 | qwen3-local | 1 | 25.4% | — | — (ref 2) | — | — | — | — | 556.30 |
| syn-en | agy-argon | 2 | 4.3% ±0.0 | 4.3% ±0.0 | 3/3 (ref 3) | 22.7% | 36.5% | 0.57 | 67.2% | 51.60 |
| syn-en | agy-argon-rescale | 1 | 4.3% | 4.3% | 3 (ref 3) | 46.4% | 57.6% | 1.09 | 48.3% | 51.40 |
| syn-en | fun-asr | 2 | 4.7% ±0.2 | 6.5% ±0.2 | 3/3 (ref 3) | 1.0% | 7.2% | 0.05 | 89.7% | 14.15 |
| syn-en | qwen3-local | 1 | 1.5% | — | — (ref 3) | — | — | — | — | 140.40 |
| syn-mix | agy-argon | 2 | 1.9% ±0.0 | 1.9% ±0.0 | 3/3 (ref 3) | 58.7% | 68.3% | 2.32 | 39.6% | 42.55 |
| syn-mix | agy-argon-rescale | 1 | 1.9% | 1.9% | 3 (ref 3) | 4.5% | 14.6% | 0.05 | 70.8% | 32.00 |
| syn-mix | fun-asr | 2 | 3.4% ±0.0 | 5.1% ±0.0 | 3/3 (ref 3) | 1.0% | 3.8% | 0.05 | 95.8% | 13.65 |
| syn-mix | qwen3-local | 1 | 5.1% | — | — (ref 3) | — | — | — | — | 112.40 |
| syn-zh | agy-argon | 2 | 0.1% ±0.0 | 0.1% ±0.0 | 3/3 (ref 3) | 35.3% | 45.3% | 1.17 | 50.0% | 52.05 |
| syn-zh | agy-argon-rescale | 1 | 0.1% | 0.1% | 3 (ref 3) | 6.9% | 11.5% | 0.04 | 81.5% | 40.20 |
| syn-zh | fun-asr | 2 | 1.7% ±0.1 | 11.8% ±0.1 | 3/3 (ref 3) | 5.0% | 7.3% | 0.07 | 66.7% | 17.45 |
| syn-zh | qwen3-local | 1 | 0.4% | — | — (ref 3) | — | — | — | — | 120.70 |

## Reading it

- **Speakers**: Fun-ASR is the only system that got the speaker count right on every sample in every run, with ≤1% wrong speaker time on real audio and 86–100% of speaker changes within 1 s. agy-argon split 2 real speakers into 14–20 labels without rescale, and found 5 of 4 on Lex #503 either way.
- **Text**: all systems are within a few points on real audio (English 9–15%, Chinese 22–25%; the floor comes from edited reference transcripts). On clean synthetic speech Gemini and local Qwen3 read slightly better than Fun-ASR (0.1–0.4% vs 1.7% CER on Chinese).
- **Stability**: Fun-ASR varies ±0.1 points between runs; agy-argon up to ±6 points and its speaker count changes run to run.
- **Speed** (15 min of audio): Fun-ASR ~30 s, agy ~2 min, local Qwen3 ~10 min.
- **Rescale**: helped agy on 4 of 5 samples (worse on syn-en) from a single run, so it is on by default but weakly supported.

Limits: one or two runs per cell, five samples, two real podcasts, no heavily code-switched real podcast yet.
