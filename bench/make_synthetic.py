"""Synthesize the three scripted benchmark conversations with Kokoro TTS.

Run with the TTS venv (kokoro + misaki[zh,en] + soundfile), not the project venv:

    VIRTUAL_ENV=~/AIContent/bench/.venv-tts ~/AIContent/bench/.venv-tts/bin/python bench/make_synthetic.py ~/AIContent/bench/samples

Each sample gets <id>.wav (16 kHz mono) and <id>.json with exact ground truth:
every utterance's speaker, text and start/end in seconds.
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

SR_TTS = 24000
SR_OUT = 16000

ZH = [
    ("A", "大家好，欢迎收听《周末慢谈》，我是主持人林晓。今天我们请到了两位嘉宾，来聊一聊年轻人为什么越来越喜欢去县城生活。"),
    ("B", "大家好，我是周明，在浙江的一个县城开了一家咖啡馆，已经第四年了。"),
    ("C", "大家好，我叫陈静，以前在上海做了八年的广告，前年搬回了老家。"),
    ("A", "周明，你当时为什么会选择离开杭州？"),
    ("B", "其实原因挺简单的。我在杭州租房子，每个月房租就要四千五，再加上通勤，每天来回差不多两个小时。"),
    ("B", "有一天晚上我加完班，站在地铁里，突然就觉得，这样的日子好像一眼就能看到头。"),
    ("A", "嗯。"),
    ("B", "所以我就算了一笔账。回老家开店，房租一年才三万多，就算赚得少一点，生活质量反而更高。"),
    ("C", "这个我特别有共鸣。我在上海的时候，工资看起来不低，但是存不下钱。"),
    ("A", "那陈静，你回去以后，最不适应的是什么？"),
    ("C", "最不适应的，其实是人情社会。在大城市，邻居之间基本不打招呼；但是在县城，你家里发生什么事情，第二天整条街都知道了。"),
    ("A", "哈哈，这个确实。"),
    ("C", "一开始我觉得很烦，后来慢慢发现，这种关系也有它温暖的地方。比如我妈生病住院那次，邻居们轮流来帮忙。"),
    ("B", "对，县城的好处就是，你遇到困难的时候，总能找到人帮你。"),
    ("A", "不过也有人说，县城的机会太少了，年轻人回去以后，很难找到合适的工作。你们怎么看？"),
    ("B", "这个要分情况。如果你是做传统行业的，确实不容易。但是现在很多工作可以远程做，比如设计、写作、做自媒体。"),
    ("C", "我现在就是一边帮以前的客户做策划，一边在县城开了一个小小的工作室，收入大概是以前的六成。"),
    ("A", "六成，那你觉得值得吗？"),
    ("C", "值得。因为我的开销降到了以前的三分之一，而且每天都能陪家人吃晚饭。"),
    ("A", "周明呢，咖啡馆在县城能活下来吗？很多人会觉得，县城的人不太喝咖啡。"),
    ("B", "第一年确实很难，每天也就卖二三十杯。后来我们开始做烘焙，又办了读书会和电影放映，慢慢就有了一群固定的客人。"),
    ("B", "现在周末一天能卖到两百杯左右，基本上能养活四个员工。"),
    ("A", "那挺不错的。"),
    ("C", "我觉得他的店已经变成我们县城年轻人的一个聚会点了。"),
    ("A", "最后一个问题，如果有听众也在考虑回县城，你们会给他们什么建议？"),
    ("B", "我的建议是，先别急着辞职。可以先回去住一两个月，看看自己能不能适应那里的节奏。"),
    ("C", "还有就是，一定要想清楚自己的收入从哪里来。情怀很重要，但是房租和饭钱更重要。"),
    ("A", "好的，非常感谢两位今天的分享。我们下期再见。"),
    ("B", "再见。"),
    ("C", "拜拜。"),
]

EN = [
    ("A", "Welcome back to Signal and Noise. I'm Rachel Kim, and today we're talking about something every engineering team argues about: on-call."),
    ("A", "With me are Marcus Webb, who runs infrastructure at a payments company, and Priya Shah, who leads a small platform team at a healthcare startup."),
    ("B", "Thanks for having me, Rachel."),
    ("C", "Happy to be here."),
    ("A", "Marcus, let's start with you. How many pages did your team get last month?"),
    ("B", "Honestly? Around three hundred and forty. And about two hundred of those were noise, alerts that fired and resolved themselves before anyone even looked."),
    ("A", "Two hundred. That's brutal."),
    ("B", "It was. We had one engineer who got woken up four nights in a row by the same disk space alert on a server we were about to decommission anyway."),
    ("C", "That's exactly the thing that burns people out. It's not the real incidents, it's the pointless ones."),
    ("A", "So what did you do about it?"),
    ("B", "We made a simple rule. If an alert pages a human and nobody takes an action, it gets reviewed the next morning. Either we fix the threshold, or we delete it."),
    ("C", "We do something similar, but we track it as a metric. We call it actionable page rate. Last quarter we were at forty one percent, and our goal is to get above eighty."),
    ("A", "Priya, healthcare sounds like a place where you can't just delete alerts, though."),
    ("C", "Right, and that's the tension. Some of our alerts are tied to compliance requirements. If a lab results feed stops for more than fifteen minutes, someone has to look at it, even at three in the morning."),
    ("B", "Sure."),
    ("C", "So we split our alerts into two tiers. Tier one pages immediately. Tier two goes into a queue that someone checks at the start of the business day."),
    ("A", "And how do you decide what goes in tier one?"),
    ("C", "We ask one question: if this waits until nine a.m., does a patient get hurt or does money get lost? If the answer is no, it's tier two."),
    ("B", "I like that. It's basically forcing the team to write down the cost of waiting."),
    ("A", "Let's talk about compensation for a second, because I know that's controversial. Should people get paid extra for being on call?"),
    ("B", "Yes. Absolutely. We pay a flat stipend per week, plus an hourly rate for any time spent actually working an incident outside business hours."),
    ("C", "We're a startup, so we can't pay much, but we give a recovery day. If you get paged after midnight, you take the next morning off, no questions asked."),
    ("A", "Does that actually get used?"),
    ("C", "At first, no. People felt guilty. So now the manager schedules it automatically. You don't have to ask."),
    ("B", "That's smart. Defaults matter more than policies."),
    ("A", "Okay, last question. If a listener is drowning in alerts right now, what's the one thing they should do this week?"),
    ("B", "Export last month's pages, sort them by how often they fired, and look at the top ten. I promise at least half of them can be fixed in an afternoon."),
    ("C", "And talk to the people who are on call. Ask them which alert they hate the most. They already know."),
    ("A", "Marcus, Priya, thank you both. This was great."),
    ("B", "Thanks, Rachel."),
    ("C", "Thank you."),
]

MIX = [
    ("A", "Hello 大家好，欢迎来到《硅谷夜话》，我是主持人 Annie。今天我们聊一个特别火的话题，就是 AI coding agent 到底能不能取代程序员。"),
    ("A", "我们请到了两位嘉宾，一位是在 Google 做 infra 的 Kevin，另一位是自己创业做 developer tools 的 Lily。"),
    ("B", "Hi Annie，大家好，我是 Kevin。"),
    ("C", "大家好，我是 Lily，我们公司主要做 code review 的自动化。"),
    ("A", "Kevin，你们组现在每天用 agent 写代码的比例大概有多少？"),
    ("B", "我粗略估计一下，新写的代码里面，大概有百分之四十是 agent 先出 draft，然后工程师再去 review 和改。"),
    ("A", "四十 percent，这个比例挺高的。"),
    ("B", "对，但是要注意，这些主要是 unit test、data migration 这种比较 boilerplate 的东西。真正 core 的 system design，还是人在做。"),
    ("C", "我同意。我们看了一下客户的数据，agent 写的 PR，merge rate 其实还不错，但是 revert rate 比人写的高大概两倍。"),
    ("A", "为什么 revert 会更多？"),
    ("C", "主要是 context 不够。比如说 agent 不知道某个 API 下个月要 deprecate，或者不知道这个 service 有一个很奇怪的 SLA 要求。"),
    ("B", "嗯，这个我们也遇到过。所以我们现在会把 design doc 和 on-call runbook 都放进 agent 的 context 里面。"),
    ("A", "那这样做以后效果怎么样？"),
    ("B", "revert rate 降了差不多一半。但是 token 的 cost 也涨了不少，一个比较大的 task，可能要跑十几万个 token。"),
    ("C", "所以现在很多公司开始算 ROI 了。不是说 agent 能做就一定要让它做，而是要看它省下来的 engineer time 值不值这个钱。"),
    ("A", "Lily，你觉得 junior engineer 会不会是最受冲击的？"),
    ("C", "短期来看是的。很多以前给 new grad 做的 task，现在 agent 就能完成。但是我觉得长期来看，会 review、会 debug 的人反而更值钱。"),
    ("B", "Exactly. 现在最缺的不是写代码的能力，而是判断代码对不对的能力。"),
    ("A", "那如果现在有一个在读的 CS 学生，你们会建议他学什么？"),
    ("B", "我会建议多做真实的 project，最好是有 user 的那种。因为只有在 production 里面，你才会遇到那些 agent 处理不了的 edge case。"),
    ("C", "还有就是，要学会写清楚的 spec。说实话，跟 agent 合作，本质上就是跟一个特别快但是没有 context 的同事合作。"),
    ("A", "这个比喻很好。"),
    ("B", "哈哈，对，而且这个同事从来不会说 no。"),
    ("A", "好的，今天的节目就到这里，感谢 Kevin 和 Lily，我们下期再见，bye bye。"),
    ("B", "Bye。"),
    ("C", "拜拜，大家。"),
]

SAMPLES = {
    "syn-zh": {"script": ZH, "lang": "zh", "voices": {"A": "zf_001", "B": "zm_010", "C": "zf_017"}},
    "syn-en": {"script": EN, "lang": "en", "voices": {"A": "af_heart", "B": "am_michael", "C": "bf_emma"}},
    "syn-mix": {"script": MIX, "lang": "mix", "voices": {"A": "zf_003", "B": "zm_009", "C": "zf_021"}},
}
NAMES = {
    "syn-zh": {"A": "林晓", "B": "周明", "C": "陈静"},
    "syn-en": {"A": "Rachel Kim", "B": "Marcus Webb", "C": "Priya Shah"},
    "syn-mix": {"A": "Annie", "B": "Kevin", "C": "Lily"},
}


def main(out_dir: Path) -> None:
    from kokoro import KModel, KPipeline

    out_dir.mkdir(parents=True, exist_ok=True)
    en_pipe = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
    zh_model = KModel(repo_id="hexgrad/Kokoro-82M-v1.1-zh").eval()
    en_g2p = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M-v1.1-zh", model=False)
    zh_pipe = KPipeline(
        lang_code="z",
        repo_id="hexgrad/Kokoro-82M-v1.1-zh",
        model=zh_model,
        en_callable=lambda text: next(en_g2p(text)).phonemes,
    )
    for sample_id, spec in SAMPLES.items():
        rng = random.Random(sample_id)
        pipe = en_pipe if spec["lang"] == "en" else zh_pipe
        pieces: list[np.ndarray] = []
        segments = []
        cursor = 0.0
        previous = None
        for speaker, text in spec["script"]:
            # Speaker changes get a conversational gap; the same speaker continuing gets a short breath.
            gap = rng.uniform(0.25, 0.7) if speaker != previous else rng.uniform(0.15, 0.35)
            if previous is None:
                gap = 0.5
            pieces.append(np.zeros(int(gap * SR_TTS), dtype=np.float32))
            cursor += gap
            speed = rng.uniform(0.95, 1.08)
            audio = np.concatenate([np.asarray(a, dtype=np.float32) for _, _, a in pipe(text, voice=spec["voices"][speaker], speed=speed)])
            audio = _trim(audio)
            duration = len(audio) / SR_TTS
            segments.append({"start": round(cursor, 3), "end": round(cursor + duration, 3), "speaker": speaker, "text": text})
            pieces.append(audio)
            cursor += duration
            previous = speaker
        pieces.append(np.zeros(int(0.8 * SR_TTS), dtype=np.float32))
        wave = np.concatenate(pieces)
        # Light room tone so silence is not digital zero (-45 dBFS).
        wave = wave + np.random.default_rng(0).normal(0, 10 ** (-45 / 20), len(wave)).astype(np.float32)
        raw = out_dir / f"{sample_id}.24k.wav"
        sf.write(raw, wave, SR_TTS, subtype="PCM_16")
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-ac", "1", "-ar", str(SR_OUT), "-c:a", "pcm_s16le", str(out_dir / f"{sample_id}.wav")],
            check=True,
        )
        raw.unlink()
        truth = {
            "id": sample_id,
            "kind": "synthetic",
            "language": spec["lang"],
            "duration": round(len(wave) / SR_TTS, 3),
            "timing": "exact",
            "speakers": NAMES[sample_id],
            "voices": spec["voices"],
            "segments": segments,
        }
        (out_dir / f"{sample_id}.json").write_text(json.dumps(truth, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{sample_id}: {truth['duration']:.0f}s, {len(segments)} turns")


def _trim(audio: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    loud = np.flatnonzero(np.abs(audio) > threshold)
    if loud.size == 0:
        return audio
    return audio[max(0, loud[0] - 240) : loud[-1] + 240]


if __name__ == "__main__":
    main(Path(sys.argv[1]).expanduser())
