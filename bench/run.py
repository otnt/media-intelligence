"""Run ASR systems over the benchmark samples and save raw hypotheses.

    uv run python bench/run.py ~/AIContent/bench --systems agy-argon,fun-asr --runs 2

Results land in <bench>/results/<system>/<sample>.run<k>.json and are reused
on the next invocation (delete a file to rerun it). No metadata context is
passed, so every system hears only the audio.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import time
import traceback
from pathlib import Path

from media_pipeline.asr.registry import require_provider
from media_pipeline.config import load_config
from media_pipeline.models import ASROptions

SYSTEMS = {
    # name: (asr model id, config overrides)
    "agy-argon": ("agy-gemini", {"agy.asr_model": "gemini-4-argon-medium"}),
    "agy-argon-rescale": ("agy-gemini", {"agy.asr_model": "gemini-4-argon-medium", "agy.rescale": True}),
    "agy-flash": ("agy-gemini", {"agy.asr_model": "gemini-3.8-flash-medium"}),
    "fun-asr": ("fun-asr", {}),
    "qwen3-local": ("qwen3-asr-1.7b", {}),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bench", type=Path)
    parser.add_argument("--systems", default="agy-argon,fun-asr")
    parser.add_argument("--samples", default="")
    parser.add_argument("--runs", type=int, default=1)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    bench = args.bench.expanduser()
    samples = sorted(p.stem for p in (bench / "samples").glob("*.json"))
    if args.samples:
        samples = [s for s in samples if s in args.samples.split(",")]
    base = load_config()
    for system in args.systems.split(","):
        model_id, overrides = SYSTEMS[system]
        config = copy.deepcopy(base)
        for dotted, value in overrides.items():
            section, field = dotted.split(".")
            setattr(getattr(config, section), field, value)
        out_dir = bench / "results" / system
        out_dir.mkdir(parents=True, exist_ok=True)
        for sample in samples:
            for run in range(1, args.runs + 1):
                target = out_dir / f"{sample}.run{run}.json"
                if target.exists():
                    continue
                provider = require_provider(model_id)
                bind = getattr(provider, "bind_config", None)
                if callable(bind):
                    bind(config)
                started = time.time()
                try:
                    transcript = provider.transcribe(bench / "samples" / f"{sample}.wav", ASROptions())
                except Exception as exc:
                    print(f"{system} {sample} run{run}: FAILED {exc}")
                    (out_dir / f"{sample}.run{run}.error.txt").write_text(traceback.format_exc(), encoding="utf-8")
                    continue
                elapsed = time.time() - started
                payload = {"system": system, "sample": sample, "run": run, "elapsed_sec": round(elapsed, 1), **transcript.to_dict()}
                target.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
                speakers = len({s.speaker for s in transcript.segments if s.speaker})
                print(f"{system} {sample} run{run}: {elapsed:.0f}s, {len(transcript.segments)} segs, {speakers} speakers", flush=True)


if __name__ == "__main__":
    main()
