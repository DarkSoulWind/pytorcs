# !pip install llama-cpp-python

import os
import sys

from llama_cpp import Llama

# COMPILATION
# pyinstaller --onefile --collect-binaries llama_cpp --collect-submodules llama_cpp \ --add-data "granite-3.3-2b-instruct-Q4_K_M.gguf:models" granite_test.py


def bundled_path(rel_path: str) -> str:
    base = getattr(sys, "_MEIPASS", os.path.abspath("."))
    return os.path.join(base, rel_path)


model_path = bundled_path(os.path.join("models", "granite-3.3-2b-instruct-Q4_K_M.gguf"))

llm = Llama(model_path=model_path, n_ctx=2048)

"""LAP SUMMARIES PROMPT"""

# out = llm.create_chat_completion(
#     messages=[
#         {"role": "system", "content": "You are an F1-style live commentator. Write EXACTLY one punchy sentence, max 20 words. Present tense, high energy, vivid verbs. Mention at least TWO numbers from the facts. Do NOT invent overtakes, positions, penalties, safety cars, or drivers. No emojis, no quotes, no multi-sentence output. Interpretation rules: damage_delta >= 2000 => big hit; offtrack_count >= 1 => runs wide; max_rpm >= 8000 => screaming engine."},
#         {"role": "user", "content": "Facts: lap=1, lap_time_s=18.07, max_speed_kmh=80.0, max_rpm=8723, offtrack_count=1, damage_delta=3846, fuel_used=0.13. Style: breathless, dramatic, radio-call energy."},
#     ],
#     max_tokens=60,
#     temperature=0.8,
#     top_p=0.9,
# )

"""INDIVIDUAL EVENT PROMPT"""

# out = llm.create_chat_completion(
#     messages=[
#         {
#             "role": "system",
#             "content": """
#                 You are an F1-style live commentator.
#                 Output EXACTLY one sentence, max 18 words. Present tense. High energy.
#                 Use vivid verbs (dives, rockets, snaps, clatters, wriggles, hooks).
#                 Mention at least one number (speed, gear, rpm, time, distance).
#                 Never invent: positions, overtakes, penalties, safety car, yellow flags, other cars, driver names.
#                 If confidence < 0.6, hedge lightly (“looks like”, “maybe”, “seems”).
#                 If severity >= 0.7, make it dramatic (“big moment”, “massive scare”, “heavy impact”).
#                 If event facts contradict each other, trust event_type and downplay the conflicting metric.
#             """,
#         },
#         {
#             "role": "user",
#             "content": """
#                 Event type: LAP_START
#                 Severity: 0.2, Confidence: 0.9
#                 Lap/Sector: 3 / 1
#                 Distance from start: 0 m
#                 Track position: middle
#                 Speed: 0.0 km/h
#                 RPM/Gear: 1100 rpm / gear 1
#             """,
#         },
#     ],
#     max_tokens=60,
#     temperature=0.8,
#     top_p=0.9,
# )

"""BATCHED EVENT PROMPT"""

# out = llm.create_chat_completion(
#     messages=[
#         {
#             "role": "system",
#             "content": """
#                 You are an F1-style live commentator.
#                 Output EXACTLY one sentence, max 22 words. Present tense, high energy.
#                 Mention at least one number (speed, rpm, gear or distance).
#                 If an event has confidence < 0.6, hedge (“looks like”, “might be”).
#                 If severity >= 0.7, make it dramatic (“big moment”, “massive scare”, “heavy impact”).
#                 Focus on the most dramatic event first, then the recovery/pressure.
#                 NEVER invent positions, overtakes, penalties, or other cars besides “a car behind” if given.
#             """,
#         },
#         {
#             "role": "user",
#             "content": """
#                 Burst window: t=1.08s to 3.06s (dt=1.98s), lap 1, sector 3, distance 3122.46m to 3126.63m, track_pos=0.29-0.33 (right side).
#                 Events in order (most important first, but keep chronology):
#                 SPIN severity=1.0 confidence=0.0 at t=1.08s, speed=0.01 km/h, rpm=942, gear=0
#                 UPSHIFT severity=0.2 confidence=0.7 at t=1.52s, speed=17.4 km/h, rpm=2168, gear=1, accel=48.2
#                 STRONG_ACCELERATION severity=1.0 confidence=1.0 at t=1.96s, speed=20.2 km/h, rpm=2482, gear=1, accel=6.53
#                 STRONG_ACCELERATION severity=1.0 confidence=1.0 at t=2.62s, speed=22.8 km/h, rpm=2929, gear=1, accel=7.36
#                 CAR_BEHIND_CLOSE severity=0.4 confidence=0.7 at t=3.06s, speed=26.6 km/h, rpm=4357, gear=1
#                 Task: Produce commentary.
#             """,
#         },
#     ],
#     max_tokens=60,
#     temperature=0.8,
#     top_p=0.9,
# )

# result = out["choices"][0]["message"]["content"].strip()
# print(f"{result=}")

import csv
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import random


# =========================
# Data model
# =========================


@dataclass(frozen=True)
class Event:
    event_type: str
    severity: float
    confidence: float
    timestamp_s: float
    lap: int
    sector: int
    dist_from_start_m: float
    track_pos: float
    speed_mps: float
    rpm: float
    gear: int
    longitudinal_accel_mps2: float


@dataclass(frozen=True)
class NarrativeState:
    pressure_level: float = 0.0  # 0..1
    threat_level: float = 0.0  # 0..1
    instability_recent: float = 0.0  # 0..1, decays
    momentum: int = 0  # -2..+2
    recovery_streak: int = 0
    last_dominant_event: str = ""
    last_opener: str = ""


# =========================
# CSV utils
# =========================


def _get(row: Dict[str, str], key: str, default: str = "") -> str:
    v = row.get(key, "")
    return v if v is not None and v != "" else default


def _to_float(s: str, default: float = 0.0) -> float:
    try:
        return float(s)
    except Exception:
        return default


def _to_int(s: str, default: int = 0) -> int:
    try:
        return int(float(s))
    except Exception:
        return default


def read_events_csv(path: str) -> List[Event]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV not found: {p.resolve()}")

    events: List[Event] = []
    with p.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {
            "event_type",
            "severity",
            "confidence",
            "timestamp_s",
            "lap",
            "sector",
            "dist_from_start_m",
            "track_pos",
            "speed_mps",
            "rpm",
            "gear",
            "longitudinal_accel_mps2",
        }
        missing = [c for c in required if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        for row in reader:
            et = _get(row, "event_type").strip()
            if not et:
                continue

            events.append(
                Event(
                    event_type=et,
                    severity=_to_float(_get(row, "severity"), 0.0),
                    confidence=_to_float(_get(row, "confidence"), 0.0),
                    timestamp_s=_to_float(_get(row, "timestamp_s"), 0.0),
                    lap=_to_int(_get(row, "lap"), 0),
                    sector=_to_int(_get(row, "sector"), 0),
                    dist_from_start_m=_to_float(_get(row, "dist_from_start_m"), 0.0),
                    track_pos=_to_float(_get(row, "track_pos"), 0.0),
                    speed_mps=_to_float(_get(row, "speed_mps"), 0.0),
                    rpm=_to_float(_get(row, "rpm"), 0.0),
                    gear=_to_int(_get(row, "gear"), 0),
                    longitudinal_accel_mps2=_to_float(
                        _get(row, "longitudinal_accel_mps2"), 0.0
                    ),
                )
            )

    return sorted(events, key=lambda e: e.timestamp_s)


# =========================
# Burst windowing
# =========================


def group_events_by_window(
    events: List[Event],
    window_s: float = 4.0,
    max_gap_s: float = 1.5,
    max_events_per_window: int = 25,
) -> List[List[Event]]:
    if not events:
        return []

    bursts: List[List[Event]] = []
    cur: List[Event] = [events[0]]
    start_t = events[0].timestamp_s

    for ev in events[1:]:
        gap = ev.timestamp_s - cur[-1].timestamp_s
        within_window = (ev.timestamp_s - start_t) <= window_s
        ok_gap = gap <= max_gap_s
        ok_count = len(cur) < max_events_per_window

        if within_window and ok_gap and ok_count:
            cur.append(ev)
        else:
            bursts.append(cur)
            cur = [ev]
            start_t = ev.timestamp_s

    bursts.append(cur)
    return bursts


# =========================
# Semantics layer
# =========================


INCIDENT_EVENTS = {"SPIN", "OFFTRACK", "LOCKUP", "HARD_BRAKING"}
PRESSURE_EVENTS = {"BEING_OVERTAKEN", "CAR_AHEAD_CLOSE"}
RECOVERY_EVENTS = {"STRONG_ACCELERATION", "UPSHIFT"}
THREAT_EVENTS = {"BEING_OVERTAKEN", "CAR_AHEAD_CLOSE"}

DRAMA_PHRASES = ("big moment", "massive scare")
HEDGE_PHRASES = ("looks like", "might have")


def _unique_preserve(xs: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def event_tags(ev: Event) -> List[str]:
    et = ev.event_type.upper()
    tags: List[str] = [et.lower().replace("_", " ")]

    if ev.confidence < 0.35:
        tags.append("very low confidence")
    elif ev.confidence < 0.60:
        tags.append("low confidence")

    if ev.severity >= 0.85:
        tags.append("very high severity")
    elif ev.severity >= 0.70:
        tags.append("high severity")

    if et in PRESSURE_EVENTS:
        tags.append("pressure")

    if et in RECOVERY_EVENTS:
        tags.append("fightback")

    if et in INCIDENT_EVENTS:
        tags.append("instability")

    if abs(ev.track_pos) > 0.8:
        tags.append("near the edge")

    return tags


def choose_headline_metric(ev: Event) -> str:
    et = ev.event_type.upper()
    speed_kmh = ev.speed_mps * 3.6

    if et in INCIDENT_EVENTS or speed_kmh < 30:
        v = int(round(speed_kmh / 10.0) * 10)
        return f"{max(v, 0)} km/h"

    r = int(round(ev.rpm / 500.0) * 500)
    if r <= 0:
        v = int(round(speed_kmh / 10.0) * 10)
        return f"{max(v, 0)} km/h"
    return f"{r} rpm"


def burst_spec(burst: List[Event]) -> Dict[str, Any]:
    dominant = max(burst, key=lambda e: (e.severity, e.timestamp_s))
    dominant_et = dominant.event_type.upper()

    confs = sorted(e.confidence for e in burst)
    median_conf = confs[len(confs) // 2] if confs else 0.0

    # Hedge rule improved: hedge only if the whole window is shaky.
    needs_hedge = (dominant.confidence < 0.35) and (median_conf < 0.50)
    soft_hedge = (not needs_hedge) and (median_conf < 0.60)

    incident_like = dominant_et in INCIDENT_EVENTS
    needs_drama = incident_like and (dominant.severity >= 0.70)

    pressure = any(e.event_type.upper() in PRESSURE_EVENTS for e in burst)
    threat = any(e.event_type.upper() in THREAT_EVENTS for e in burst)
    recovery = any(e.event_type.upper() in RECOVERY_EVENTS for e in burst)
    instability = any(e.event_type.upper() in INCIDENT_EVENTS for e in burst)

    tags: List[str] = []
    for ev in burst:
        tags.extend(event_tags(ev))
    tags = _unique_preserve(tags)

    headline = choose_headline_metric(dominant)
    types_seq = [e.event_type.upper() for e in burst]
    compact_seq = " > ".join(_unique_preserve(types_seq)[:6])

    return {
        "dominant_event": dominant_et,
        "needs_hedge": needs_hedge,
        "soft_hedge": soft_hedge,
        "needs_drama": needs_drama,
        "pressure": pressure,
        "threat": threat,
        "recovery": recovery,
        "instability": instability,
        "headline": headline,
        "tags": tags[:12],
        "sequence": compact_seq,
        "count": len(burst),
    }


# =========================
# Narrative state update
# =========================


def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def update_state(state: NarrativeState, spec: Dict[str, Any]) -> NarrativeState:
    pressure = 1.0 if spec["pressure"] else 0.0
    threat = 1.0 if spec["threat"] else 0.0
    instability = 1.0 if spec["instability"] else 0.0
    recovery = 1.0 if spec["recovery"] else 0.0

    new_pressure = clamp01(
        state.pressure_level * 0.78 + 0.35 * pressure + 0.10 * threat
    )
    new_threat = clamp01(state.threat_level * 0.80 + 0.40 * threat)
    new_instability = clamp01(state.instability_recent * 0.72 + 0.55 * instability)

    mom = state.momentum
    if instability:
        mom -= 1
    if recovery and (pressure or threat):
        mom += 1
    mom = max(-2, min(2, mom))

    rec_streak = state.recovery_streak
    if recovery and not instability:
        rec_streak += 1
    else:
        rec_streak = max(0, rec_streak - 1)

    return replace(
        state,
        pressure_level=new_pressure,
        threat_level=new_threat,
        instability_recent=new_instability,
        momentum=mom,
        recovery_streak=rec_streak,
        last_dominant_event=str(spec["dominant_event"]),
    )


def state_tags(state: NarrativeState) -> List[str]:
    tags: List[str] = []
    if state.instability_recent >= 0.55:
        tags.append("shaky")
    if state.threat_level >= 0.55:
        tags.append("under threat")
    if state.pressure_level >= 0.55:
        tags.append("under pressure")
    if state.momentum <= -1:
        tags.append("on the back foot")
    if state.momentum >= 1:
        tags.append("finding rhythm")
    if state.recovery_streak >= 2:
        tags.append("fightback building")
    return tags


# =========================
# EA-ish line library (upgraded)
# =========================

LINE_LIBRARY: Dict[str, List[str]] = {
    "SPIN": [
        "{HEDGE} the rear snaps loose, the car rotates; {DRAMA} at {METRIC}.",
        "{HEDGE} big slide, catches it late; {DRAMA} at {METRIC}.",
        "{HEDGE} the car loops it, scrabbles for grip; {DRAMA} at {METRIC}.",
    ],
    "HARD_BRAKING": [
        "{HEDGE} late on the brakes, a twitch on entry; gathers it at {METRIC}.",
        "{HEDGE} heavy braking, front tyres protest; still makes it at {METRIC}.",
    ],
    "LOCKUP": [
        "{HEDGE} locks up into the corner, smoke and squeal; holds it at {METRIC}.",
        "{HEDGE} brief lockup, runs it tight; keeps control at {METRIC}.",
    ],
    "OFFTRACK": [
        "{HEDGE} runs wide to the edge, dusty line; rescues it at {METRIC}.",
        "{HEDGE} near the edge, skips over the dirt; survives at {METRIC}.",
    ],
    "BEING_OVERTAKEN": [
        "{HEDGE} under attack, moves once and holds firm; just hangs on at {METRIC}.",
        "{HEDGE} squeezed hard, defends cleanly; keeps it straight at {METRIC}.",
        "{HEDGE} threat looming, traction scrappy; stays in it at {METRIC}.",
    ],
    "CAR_AHEAD_CLOSE": [
        "{HEDGE} right on the tail, the car wriggles; pressure rises at {METRIC}.",
        "{HEDGE} closing rapidly, commits late; keeps it tidy at {METRIC}.",
        "{HEDGE} nose-to-tail, tiny wobble mid-corner; stays composed at {METRIC}.",
    ],
    "STRONG_ACCELERATION": [
        "{HEDGE} traction bites, the car fires out; builds momentum at {METRIC}.",
        "{HEDGE} clean exit, gets it rotated early; surges at {METRIC}.",
        "{HEDGE} hooks up the power, straightens it fast; charges at {METRIC}.",
    ],
    "UPSHIFT": [
        "{HEDGE} clicks up cleanly, keeps it balanced; settles at {METRIC}.",
        "{HEDGE} short-shifts to calm it down; stays neat at {METRIC}.",
    ],
}

FALLBACK_LIBRARY = [
    "{HEDGE} tense moment, the car keeps it tidy; holds on at {METRIC}.",
    "{HEDGE} scrappy phase, fighting the balance; survives at {METRIC}.",
    "{HEDGE} pressure builds, tiny correction; stays straight at {METRIC}.",
]


def pick_skeleton(spec: Dict[str, Any], state: NarrativeState) -> str:
    dom = str(spec["dominant_event"])
    candidates = LINE_LIBRARY.get(dom, FALLBACK_LIBRARY)

    # Gentle bias using narrative state
    if state.threat_level >= 0.65 and dom not in INCIDENT_EVENTS:
        candidates = candidates + [
            "{HEDGE} still under pressure, the car hangs tough; holds it at {METRIC}.",
            "{HEDGE} threat all around, keeps it clean; survives at {METRIC}.",
        ]
    if state.instability_recent >= 0.65 and dom not in INCIDENT_EVENTS:
        candidates = candidates + [
            "{HEDGE} still shaky, the rear feels loose; calms it at {METRIC}.",
            "{HEDGE} fighting grip, tiny slide again; steadies it at {METRIC}.",
        ]
    if state.recovery_streak >= 2:
        candidates = candidates + [
            "{HEDGE} fightback building, gets traction down; pushes on at {METRIC}.",
        ]

    return random.choice(candidates)


def render_skeleton(spec: Dict[str, Any], skeleton: str) -> str:
    hedge_required = bool(spec["needs_hedge"])
    soft_hedge = bool(spec["soft_hedge"])
    drama_required = bool(spec["needs_drama"])

    hedge = ""
    if hedge_required:
        hedge = random.choice(HEDGE_PHRASES)
    elif soft_hedge and random.random() < 0.35:
        hedge = random.choice(HEDGE_PHRASES)

    drama = ""
    if drama_required:
        drama = random.choice(DRAMA_PHRASES)

    text = skeleton
    text = text.replace("{HEDGE}", (hedge + " ") if hedge else "")
    text = text.replace("{DRAMA}", drama if drama else "")
    text = text.replace("{METRIC}", str(spec["headline"]))

    text = re.sub(r"\s{2,}", " ", text).strip()
    text = re.sub(r"\s+([.;,])", r"\1", text)
    text = text.replace(" ;", ";").replace(" .", ".").replace(" ,", ",")
    return text


# =========================
# Prompting
# =========================

SYSTEM_PROMPT = """
You are an F1-style live race commentator: short, sharp, punchy.
You are rewriting a draft line into natural broadcast cadence.

Output ONE sentence only, 12–22 words, present tense.
Max TWO clauses. Periods or semicolons are fine; commas are allowed.
NO quotes. NO exclamation marks.

Use exactly ONE number only, and it MUST be the given headline metric.
Do NOT start the line with the number.

Never invent: positions, penalties, pits, fans, the pack, lap records.
No driver/team names; use "the car" only.

If Hedge required is True: include exactly one hedge phrase: "looks like" or "might have".
If Hedge required is False but Soft hedge is True: hedging is optional, do not force it.

If Drama required is True: include exactly one dramatic phrase: "big moment" or "massive scare".
If Drama required is False: do not include those phrases.
""".strip()


def build_prompt(
    spec: Dict[str, Any],
    state: NarrativeState,
    draft_line: str,
    recent_openers: List[str],
) -> str:
    avoid = ", ".join([o for o in recent_openers[-3:] if o]) or "none"
    s_tags = ", ".join(state_tags(state)) or "neutral"

    return (
        f"Window summary: {spec['sequence']} (events={spec['count']})\n"
        f"Context tags: {', '.join(spec['tags'])}\n"
        f"Narrative state: {s_tags}\n"
        f"Dominant event: {spec['dominant_event']}\n"
        f"Headline metric (ONLY number): {spec['headline']}\n"
        f"Hedge required: {spec['needs_hedge']}\n"
        f"Soft hedge: {spec['soft_hedge']}\n"
        f"Drama required: {spec['needs_drama']}\n"
        f"Avoid reusing these opening words: {avoid}\n"
        f"Draft line to rewrite:\n{draft_line}\n"
        "Task: Rewrite the draft into natural F1 cadence while obeying every rule."
    )


# =========================
# Validation + anti-repetition
# =========================


def _words(text: str) -> List[str]:
    return re.findall(r"\b[\w']+\b", text.lower())


def line_opener(text: str, n: int = 3) -> str:
    w = _words(text)
    return " ".join(w[:n]) if w else ""


def validate_line(
    text: str,
    spec: Dict[str, Any],
    recent_lines: List[str],
) -> Tuple[bool, str]:
    t = text.strip()

    # Hard bans (commas allowed now)
    if any(q in t for q in ['"', "“", "”", "’", "‘"]):
        return False, "contains quotes"
    if "!" in t:
        return False, "contains exclamation"

    # One sentence-ish: allow at most one period; semicolons ok
    if t.count(".") > 1:
        return False, "too many sentences"

    # Word count: wider so it can breathe
    w = re.findall(r"\b[\w']+\b", t)
    if not (12 <= len(w) <= 22):
        return False, f"word count {len(w)}"

    # Exactly one number
    nums = re.findall(r"\d+(?:\.\d+)?", t)
    if len(nums) != 1:
        return False, f"number count {len(nums)}"

    # Must include the headline metric verbatim
    headline = str(spec["headline"])
    if headline not in t:
        return False, "missing headline metric"

    # Must not start with a number
    if re.match(r"^\s*\d", t):
        return False, "starts with a number"

    # Hedge rules
    needs_hedge = bool(spec["needs_hedge"])
    soft_hedge = bool(spec["soft_hedge"])
    has_looks = "looks like" in t.lower()
    has_might = "might have" in t.lower()
    hedge_count = int(has_looks) + int(has_might)

    if needs_hedge:
        if hedge_count != 1:
            return False, "hedge required but missing or multiple"
    else:
        if not soft_hedge and hedge_count > 0:
            return False, "hedge used when not allowed"

    # Drama rules
    needs_drama = bool(spec["needs_drama"])
    drama_hits = sum(1 for p in DRAMA_PHRASES if p in t.lower())
    if needs_drama:
        if drama_hits != 1:
            return False, "drama required but missing or multiple"
    else:
        if drama_hits > 0:
            return False, "drama used when not allowed"

    # Anti-repetition: opening pattern
    opener = line_opener(t, 3)
    recent_openers = [line_opener(x, 3) for x in recent_lines[-3:]]
    if opener and opener in recent_openers:
        return False, "repeated opener"

    # Phrase fatigue bans (stops your “under threat; hangs on…” spam)
    fatigue_phrases = [
        "under threat",
        "hangs on",
        "tries to calm",
        "closing fast",
        "pressure rises",
        "stays composed",
        "keeps it tidy",
    ]
    last3 = " ".join(recent_lines[-3:]).lower()
    for p in fatigue_phrases:
        if p in t.lower() and p in last3:
            return False, f"phrase fatigue: {p}"

    return True, "ok"


# =========================
# Generation
# =========================


def llm_rewrite(llm, prompt: str, temperature: float = 0.65) -> str:
    out = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=90,
        temperature=temperature,
        top_p=0.9,
        repeat_penalty=1.12,
        frequency_penalty=0.20,
        presence_penalty=0.12,
    )
    return out["choices"][0]["message"]["content"].strip()


def generate_line(
    llm,
    spec: Dict[str, Any],
    state: NarrativeState,
    recent_lines: List[str],
    recent_openers: List[str],
    max_retries: int = 8,
) -> str:
    skeleton = pick_skeleton(spec, state)
    draft = render_skeleton(spec, skeleton)

    if llm is None:
        return draft

    prompt = build_prompt(spec, state, draft, recent_openers)
    last_text = draft

    for attempt in range(max_retries):
        temp = 0.60 + 0.05 * min(attempt, 3)
        last_text = llm_rewrite(llm, prompt, temperature=temp)

        ok, reason = validate_line(last_text, spec, recent_lines)
        if ok:
            return last_text

        prompt = (
            build_prompt(spec, state, draft, recent_openers)
            + f"\nYour last output was invalid: {reason}.\n"
            "Rewrite it to satisfy ALL rules exactly.\n"
            "Reminder: one sentence; 12–22 words; no quotes; no exclamation; "
            "do not start with the number; use the headline metric as the ONLY number."
        )

    return draft


# =========================
# Pipeline
# =========================


def run_pipeline(
    llm,
    csv_path: str = "events.csv",
    out_path: str = "commentary.txt",
    window_s: float = 4.0,
    max_gap_s: float = 1.5,
    seed: int = 7,
) -> List[str]:
    random.seed(seed)

    events = read_events_csv(csv_path)
    bursts = group_events_by_window(events, window_s=window_s, max_gap_s=max_gap_s)

    outputs: List[str] = []
    openers: List[str] = []
    state = NarrativeState()

    for burst in bursts:
        spec = burst_spec(burst)
        state = update_state(state, spec)

        line = generate_line(
            llm=llm,
            spec=spec,
            state=state,
            recent_lines=outputs,
            recent_openers=openers,
        )

        outputs.append(line)
        openers.append(line_opener(line, 3))
        state = replace(state, last_opener=openers[-1])

    Path(out_path).write_text("\n".join(outputs), encoding="utf-8")
    return outputs


if __name__ == "__main__":
    # Hook up your llm instance here.
    # from llama_cpp import Llama
    # llm = Llama.from_pretrained(...)
    run_pipeline(llm, "events.csv", "commentary.txt")
    raise SystemExit("Hook up llm and run run_pipeline().")
