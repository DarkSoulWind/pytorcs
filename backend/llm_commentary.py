# !pip install llama-cpp-python

import os
import sys
import csv
import re
import random
from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass, replace
from typing import Dict, List, Tuple, Optional, Any, Iterator

from llama_cpp import Llama


# =========================
# Bundled path helper
# =========================


def bundled_path(rel_path: str) -> str:
    base = getattr(sys, "_MEIPASS", os.path.abspath("."))
    return os.path.join(base, rel_path)


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
    pressure_level: float = 0.0
    threat_level: float = 0.0
    instability_recent: float = 0.0
    momentum: int = 0
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
# Style bank (retrieval)
# =========================


def read_style_bank_csv(path: str) -> Dict[str, List[str]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Style bank CSV not found: {p.resolve()}")

    bank: Dict[str, List[str]] = {}
    with p.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("Style bank CSV missing headers.")
        if (
            "event_type" not in reader.fieldnames
            or "commentary" not in reader.fieldnames
        ):
            raise ValueError("Style bank CSV must have columns: event_type,commentary")

        for row in reader:
            et = (_get(row, "event_type").strip() or "").upper()
            line = _get(row, "commentary").strip() or ""
            if not et or not line:
                continue
            bank.setdefault(et, []).append(line)

    for k, v in bank.items():
        seen = set()
        out = []
        for s in v:
            s = s.strip()
            if s and s not in seen:
                seen.add(s)
                out.append(s)
        bank[k] = out

    return bank


def sample_style_examples(
    style_bank: Dict[str, List[str]],
    dominant_event: str,
    k: int = 8,
    seed_key: Optional[str] = None,
) -> List[str]:
    dom = (dominant_event or "").upper()
    pool = style_bank.get(dom, [])[:]
    if not pool:
        pool = style_bank.get("GENERAL", [])[:]
    if not pool:
        for vv in style_bank.values():
            pool.extend(vv)

    if not pool:
        return []

    rng = random.Random()
    if seed_key is not None:
        rng.seed(seed_key)

    seen = set()
    deduped = []
    for s in pool:
        s2 = s.strip()
        if s2 and s2 not in seen:
            seen.add(s2)
            deduped.append(s2)

    if len(deduped) <= k:
        return deduped
    return rng.sample(deduped, k)


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
# Semantics
# =========================

INCIDENT_EVENTS = {"SPIN", "OFFTRACK", "LOCKUP", "HARD_BRAKING"}
PRESSURE_EVENTS = {"BEING_OVERTAKEN", "CAR_AHEAD_CLOSE"}
RECOVERY_EVENTS = {"STRONG_ACCELERATION", "UPSHIFT"}
THREAT_EVENTS = {"BEING_OVERTAKEN", "CAR_AHEAD_CLOSE"}
HIGH_DRAMA_EVENTS = {"SPIN", "WALL_IMPACT", "CONTACT", "DAMAGE_EVENT"}
URGENT_EVENTS = {
    "SPIN",
    "SLIDE",
    "OFFTRACK",
    "LOCKUP",
    "HARD_BRAKING",
    "WALL_IMPACT",
}

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


def choose_headline_metric(
    burst: List[Event], dominant: Event
) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns (metric, metric_reason).
    metric_reason is one of: rpm_spike, rpm_limiter, rpm_defines_moment,
    speed_headline, speed_shockingly_low, speed_braking_commitment.
    """
    if not burst:
        return None, None

    et = dominant.event_type.upper()
    speed_kmh = max(dominant.speed_mps * 3.6, 0.0)
    rpms = [e.rpm for e in burst if e.rpm > 0]
    speeds = [max(e.speed_mps * 3.6, 0.0) for e in burst]

    burst_peak_rpm = max(rpms) if rpms else 0.0
    burst_peak_speed = max(speeds) if speeds else speed_kmh
    burst_min_speed = min(speeds) if speeds else speed_kmh
    burst_avg_rpm = (sum(rpms) / len(rpms)) if rpms else 0.0

    severe = dominant.severity >= 0.70
    heavy_brake = dominant.longitudinal_accel_mps2 <= -5.5

    rpm_spike = (
        dominant.rpm > 0
        and burst_avg_rpm > 0
        and dominant.rpm >= burst_avg_rpm + 1200.0
        and dominant.rpm >= 3500.0
    )
    rpm_limiter = (
        dominant.rpm >= 0.96 * max(burst_peak_rpm, 1.0) and dominant.rpm >= 7000.0
    )
    rpm_defines_moment = (
        et in {"STRONG_ACCELERATION", "UPSHIFT"} and severe and dominant.rpm >= 3500.0
    )

    speed_headline = et in INCIDENT_EVENTS and speed_kmh >= 120.0
    speed_shockingly_low = et in INCIDENT_EVENTS and burst_min_speed <= 45.0
    speed_braking_commitment = (
        et in {"HARD_BRAKING", "LOCKUP"}
        and speed_kmh >= 70.0
        and (heavy_brake or severe)
    )

    if rpm_limiter:
        r = int(round(dominant.rpm / 500.0) * 500)
        if r > 0:
            return f"{r} RPM", "rpm_limiter"
    if rpm_spike:
        r = int(round(dominant.rpm / 500.0) * 500)
        if r > 0:
            return f"{r} RPM", "rpm_spike"
    if rpm_defines_moment:
        r = int(round(dominant.rpm / 500.0) * 500)
        if r > 0:
            return f"{r} RPM", "rpm_defines_moment"

    if speed_shockingly_low:
        v = int(round(burst_min_speed / 10.0) * 10)
        return f"{max(v, 0)} km/h", "speed_shockingly_low"
    if speed_braking_commitment:
        v = int(round(speed_kmh / 10.0) * 10)
        return f"{max(v, 0)} km/h", "speed_braking_commitment"
    if speed_headline:
        v = int(round(burst_peak_speed / 10.0) * 10)
        return f"{max(v, 0)} km/h", "speed_headline"

    return None, None


def classify_intensity_tone(dominant_event: str, severity: float) -> str:
    et = (dominant_event or "").upper()
    sev = float(severity)
    if et in HIGH_DRAMA_EVENTS and sev >= 0.88:
        return "dramatic"
    if et in URGENT_EVENTS and sev >= 0.68:
        return "urgent"
    return "calm"


def burst_spec(burst: List[Event]) -> Dict[str, Any]:
    dominant = max(burst, key=lambda e: (e.severity, e.timestamp_s))
    dominant_et = dominant.event_type.upper()

    confs = sorted(e.confidence for e in burst)
    median_conf = confs[len(confs) // 2] if confs else 0.0

    needs_hedge = (dominant.confidence < 0.35) and (median_conf < 0.50)
    soft_hedge = (not needs_hedge) and (median_conf < 0.60)

    tone_level = classify_intensity_tone(dominant_et, float(dominant.severity))
    needs_drama = tone_level == "dramatic"

    pressure = any(e.event_type.upper() in PRESSURE_EVENTS for e in burst)
    threat = any(e.event_type.upper() in THREAT_EVENTS for e in burst)
    recovery = any(e.event_type.upper() in RECOVERY_EVENTS for e in burst)
    instability = any(e.event_type.upper() in INCIDENT_EVENTS for e in burst)

    tags: List[str] = []
    for ev in burst:
        tags.extend(event_tags(ev))
    tags = _unique_preserve(tags)

    headline, metric_reason = choose_headline_metric(burst, dominant)
    types_seq = [e.event_type.upper() for e in burst]
    compact_seq = " > ".join(_unique_preserve(types_seq)[:6])

    intensity_peak = tone_level == "dramatic"

    return {
        "dominant_event": dominant_et,
        "timestamp_s": float(dominant.timestamp_s),
        "dominant_severity": float(dominant.severity),
        "tone_level": tone_level,
        "intensity_peak": intensity_peak,
        "needs_hedge": needs_hedge,
        "soft_hedge": soft_hedge,
        "needs_drama": needs_drama,
        "pressure": pressure,
        "threat": threat,
        "recovery": recovery,
        "instability": instability,
        "headline": headline,
        "metric_reason": metric_reason,
        "has_metric": bool(headline),
        "tags": tags[:12],
        "sequence": compact_seq,
        "count": len(burst),
    }


def should_shout_line(spec: Dict[str, Any]) -> bool:
    return str(spec.get("tone_level", "calm")) == "dramatic"


def emphasize_intensity_clause(text: str, spec: Dict[str, Any]) -> str:
    if not should_shout_line(spec):
        return text.strip()

    line = text.strip()
    idx = line.find(",")
    if idx <= 0:
        return line

    lead = line[:idx].strip()
    rest = line[idx + 1 :].strip()
    wc = len(re.findall(r"\b[\w']+\b", lead))
    if 2 <= wc <= 8 and not lead.isupper():
        if not rest:
            return f"{lead.upper()}!"
        return f"{lead.upper()}! {rest}"
    return line


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
# Realism controls
# =========================

BANNED_TERMS = {
    "pack",
    "grid",
    "midfield",
    "title fight",
    "championship",
    "parc fermé",
    "shake-up",
    "fans",
    "grand prix",
    "drs",
    "pit",
    "penalty",
    "record",
    "safety car",
    "lap record",
    "sector",
    "pilot",
}

BANNED_PURPLE = {
    "spirits",
    "dampened",
    "relentless",
    "composure",
    "tense dance",
    "navigates",
    "poised",
    "valiant",
    "heroic",
    "symphony",
    "ballet",
    "poetry",
    "glory",
    "destiny",
}

# Words/phrases that keep making your output feel like a stamp.
TIRED_PHRASES = {
    "and it's a clean acceleration",
    "small correction on throttle",
    "rear grips up",
    "builds speed",
    "carries on",
    "clean and straight",
    "stays on throttle smoothly",
    "gets a moment",
    "keeps it together",
}

# Verbs allowed as openers (after optional hedge)
VERB_OPENERS = {
    "locks",
    "locked",
    "slides",
    "spins",
    "runs",
    "goes",
    "defends",
    "covers",
    "fires",
    "hooks",
    "dives",
    "catches",
    "gathers",
    "snaps",
    "brakes",
    "surges",
    "drops",
    "skates",
    "skips",
    "checks",
    "saves",
    "loses",
    "hits",
    "stamps",
    "leans",
    "closes",
    "sits",
    "fills",
    "leaves",
    "launches",
    "drives",
    "gets",
    "steps",
    "straightens",
}


def _recent_contains(recent_lines: List[str], needle: str, n: int = 10) -> bool:
    blob = " ".join(recent_lines[-n:]).lower()
    return needle.lower() in blob


def pick_fresh(
    pool: List[str], recent_lines: List[str], n: int = 10, tries: int = 16
) -> str:
    if not pool:
        return ""
    for _ in range(tries):
        c = random.choice(pool)
        if not _recent_contains(recent_lines, c, n=n):
            return c
    return random.choice(pool)


def pick_action_with_opener_cooldown(
    pool: List[str], recent_lines: List[str], cooldown: int = 14
) -> str:
    if not pool:
        return ""
    recent_roots = {
        line_opener_key(line, 2) for line in recent_lines[-cooldown:] if line.strip()
    }
    fresh = [a for a in pool if line_opener_key(a, 2) not in recent_roots]
    if fresh:
        return random.choice(fresh)
    # If all opener roots are exhausted, still prefer text not seen recently.
    return pick_fresh(pool, recent_lines, n=max(cooldown, 10), tries=24)


# =========================
# Draft generator (varied cadence, metric-aware)
# =========================


def metric_is_kmh(metric: str) -> bool:
    return bool(metric) and "km/h" in metric.lower()


EVENT_ACTIONS = {
    "LOCKUP": [
        "Locks a front",
        "Locks up on entry",
        "Checks up under braking",
        "Runs deep under braking",
        "Gets it stopped late",
    ],
    "HARD_BRAKING": [
        "Hits the anchors",
        "Brakes very late",
        "Stamps on the pedal",
        "Hard on the brakes",
        "Big stop into the corner",
    ],
    "SPIN": [
        "Rear snaps",
        "Loses the rear",
        "Round it goes",
        "Steps out sharply",
        "That’s a half-spin",
    ],
    "OFFTRACK": [
        "Runs wide",
        "Drops a wheel",
        "Skates to the edge",
        "Runs out of road",
        "Clips the dirt",
    ],
    "BEING_OVERTAKEN": [
        "Defends the inside",
        "Covers it off",
        "Leaves no space",
        "Holds firm",
        "Squeezed on entry",
    ],
    "CAR_AHEAD_CLOSE": [
        "Closes right up",
        "Sits on the rear",
        "Nose-to-tail",
        "Fills the mirrors",
        "Gets tucked in",
    ],
    "STRONG_ACCELERATION": [
        "Gets the power down",
        "Drives off cleanly",
        "Straightens it early",
        "Fires out",
        "Launches out",
    ],
    "UPSHIFT": [
        "Clicks up cleanly",
        "Short-shifts",
        "Takes the next gear",
        "Keeps it smooth",
        "Feeds the throttle in",
    ],
    "DEFAULT": [
        "Catches a small slide",
        "Checks the rear once",
        "Makes a quick correction",
        "Handles a twitch",
        "Settles the car down",
    ],
}

CONS_KMH = {
    "LOCKUP": [
        "fronts chirping",
        "nose washing wide",
        "tiny puff of smoke",
        "runs a touch long",
        "nearly misses the apex",
    ],
    "HARD_BRAKING": [
        "front end skating",
        "rear getting light",
        "a twitch mid-stop",
        "right on the limit",
        "brakes complaining",
    ],
    "SPIN": [
        "rotation starts instantly",
        "grip just disappears",
        "it swaps ends",
        "rotating fast",
        "sliding broadside",
    ],
    "OFFTRACK": [
        "on the dusty line",
        "grip compromised",
        "dirt on the tyres",
        "runs onto the dirt",
        "right out to the edge",
    ],
    "DEFAULT": [
        "a twitch on entry",
        "a small correction",
        "a moment of oversteer",
        "right on the kerb",
    ],
}

CONS_RPM = {
    "STRONG_ACCELERATION": [
        "rear hooks up",
        "bites on exit",
        "power comes in early",
        "clean on throttle",
        "gets traction down",
    ],
    "UPSHIFT": [
        "keeps it balanced",
        "settles the rear",
        "smooth on the change",
        "no drama on exit",
        "stays composed",
    ],
    "DEFAULT": [
        "engine cleans up",
        "rear moves slightly",
        "small correction on throttle",
        "keeps it tidy",
    ],
}

OUTCOMES = {
    "SPIN": [
        "stays running",
        "gathers it up",
        "loses a little time",
        "avoids anything worse",
    ],
    "LOCKUP": [
        "keeps it straight",
        "still makes the corner",
        "avoids the worst of it",
        "recovers on exit",
    ],
    "HARD_BRAKING": [
        "makes the corner",
        "holds together",
        "recovers on exit",
        "keeps it tidy",
    ],
    "OFFTRACK": [
        "pulls it back on line",
        "gets away with it",
        "steadies it up",
        "keeps it running",
    ],
    "STRONG_ACCELERATION": [
        "builds speed",
        "goes again",
        "settles quickly",
        "pushes on",
    ],
    "UPSHIFT": [
        "settles down",
        "stays smooth",
        "keeps momentum",
        "looks calmer now",
    ],
    "DEFAULT": [
        "carries on",
        "keeps it together",
        "steadies up",
        "stays under control",
    ],
}

# More cadence patterns = less “generated” smell.
# No semicolons. Ever.
PATTERNS_KMH = [
    "{A} at {M}, {C}, {O}.",
    "{A}, {C} at {M}, {O}.",
    "{A} at {M} with {C}, {O}.",
    "{A}, {O} at {M}, {C}.",
    "{A} at {M}, {O}, {C}.",
]
PATTERNS_RPM = [
    "{A} at {M}, {C}, {O}.",
    "{A}, {C} at {M}, {O}.",
    "{A} at {M}, {O}, {C}.",
    "{A} at {M} as the rear {C}, {O}.",
    "{A}, {O} at {M}, {C}.",
]
PATTERNS_NO_METRIC = [
    "{A}, {C}, {O}.",
    "{A}, {O}, {C}.",
    "{A}, {C}, and {O}.",
    "{A}, {O}, then {C}.",
]
PATTERNS_SHORT_KMH = [
    "{A} at {M}, {O}.",
    "{A} at {M}, {C}.",
    "{A}, {C} at {M}.",
]
PATTERNS_SHORT_RPM = [
    "{A} at {M}, {O}.",
    "{A} at {M}, {C}.",
    "{A}, {C} at {M}.",
]
PATTERNS_SHORT_NO_METRIC = [
    "{A}, {O}.",
    "{A}, {C}.",
    "{A}, then {O}.",
]


def make_beat_plan(spec: Dict[str, Any], recent_lines: List[str]) -> str:
    dom = str(spec["dominant_event"]).upper()
    metric = str(spec.get("headline") or "")
    has_metric = bool(spec.get("has_metric")) and bool(metric)
    event_count = int(spec.get("count", 0))

    action_pool = EVENT_ACTIONS.get(dom, EVENT_ACTIONS["DEFAULT"])
    action = pick_action_with_opener_cooldown(action_pool, recent_lines, cooldown=14)
    outcome = pick_fresh(
        OUTCOMES.get(dom, OUTCOMES["DEFAULT"]), recent_lines, n=14, tries=24
    )
    short_mode = event_count <= 2

    def _phrase_conflict(a: str, b: str) -> bool:
        al = a.lower()
        bl = b.lower()
        location_markers = ("on entry", "on exit", "under braking", "on throttle")
        if any(m in al and m in bl for m in location_markers):
            return True
        a_words = set(re.findall(r"[a-z']+", al))
        b_words = set(re.findall(r"[a-z']+", bl))
        stop = {
            "a",
            "an",
            "the",
            "it",
            "and",
            "or",
            "to",
            "of",
            "on",
            "in",
            "with",
            "at",
            "for",
            "up",
            "down",
            "then",
        }
        overlap = (a_words - stop) & (b_words - stop)
        return len(overlap) >= 2

    if has_metric and metric_is_kmh(metric):
        cons_pool = CONS_KMH.get(dom, CONS_KMH["DEFAULT"])
        cons = pick_fresh(cons_pool, recent_lines, n=14, tries=24)
        if _phrase_conflict(action, cons):
            alt = [
                c for c in cons_pool if c != cons and not _phrase_conflict(action, c)
            ]
            if alt:
                cons = pick_fresh(alt, recent_lines, n=14, tries=24)
        pattern = pick_fresh(
            PATTERNS_SHORT_KMH if short_mode else PATTERNS_KMH,
            recent_lines,
            n=12,
            tries=24,
        )
    elif has_metric:
        cons_pool = CONS_RPM.get(dom, CONS_RPM["DEFAULT"])
        cons = pick_fresh(cons_pool, recent_lines, n=14, tries=24)
        if _phrase_conflict(action, cons):
            alt = [
                c for c in cons_pool if c != cons and not _phrase_conflict(action, c)
            ]
            if alt:
                cons = pick_fresh(alt, recent_lines, n=14, tries=24)
        pattern = pick_fresh(
            PATTERNS_SHORT_RPM if short_mode else PATTERNS_RPM,
            recent_lines,
            n=12,
            tries=24,
        )
    else:
        cons_pool = CONS_KMH.get(dom, CONS_KMH["DEFAULT"]) + CONS_RPM.get(
            dom, CONS_RPM["DEFAULT"]
        )
        cons = pick_fresh(cons_pool, recent_lines, n=14, tries=24)
        if _phrase_conflict(action, cons):
            alt = [
                c for c in cons_pool if c != cons and not _phrase_conflict(action, c)
            ]
            if alt:
                cons = pick_fresh(alt, recent_lines, n=14, tries=24)
        pattern = pick_fresh(
            PATTERNS_SHORT_NO_METRIC if short_mode else PATTERNS_NO_METRIC,
            recent_lines,
            n=12,
            tries=24,
        )

    draft = pattern.format(A=action, C=cons, M=metric, O=outcome)
    draft = re.sub(r"\s{2,}", " ", draft).strip()
    return draft


# =========================
# Prompting
# =========================

SYSTEM_PROMPT = """
You are a live motorsport commentator. Sound like real broadcast speech, not a report.

Write ONE sentence only, present tense, 8–22 words.
Use cadence based on target:
- concise: 1–2 clauses only.
- standard: 2–3 clauses.
Use commas naturally. DO NOT use semicolons. DO NOT use dashes.

If Headline metric is provided:
- Use exactly ONE number only, and it MUST be that headline metric.
- Do NOT start the line with the number.
If Headline metric is not provided:
- Use zero numbers.
- Do not mention RPM or km/h.

Never invent: positions, pit stops, penalties, safety cars, DRS, records, "the pack", "the grid".
No driver/team names.
Do not call the driver "pilot".
Never write broken conjunctions like "then and" or "and and".
Never include meta labels or production notes like "(Tone: calm)", "[urgent]", "Tone target:", or stage directions.
Output commentary only; no bracketed/parenthetical instructions.

If Hedge required is True: include exactly one hedge phrase: "looks like" or "might have".
If Hedge required is False but Soft hedge is True: hedging is optional.
If Drama required is True: include exactly one phrase: "big moment" or "massive scare".
If Drama required is False: do not include those phrases.
Tone target controls emotional scaling:
- calm: measured and controlled wording, avoid alarmist language.
- urgent: sharper and more immediate wording, but still controlled.
- dramatic: high-intensity wording with clear urgency and impact.
Do not write all lines at the same emotional level.

Avoid flowery language. Avoid awkward fragments. Avoid repeating "it".
Limit "and it" to at most ONE time.
Do not say "and it's a clean acceleration" or other unnatural summaries.
Avoid repeating the same location fragment twice in one line (e.g., "on entry" + "on entry").
Do not reuse an opening clause used in recent lines.
Avoid generic repetition like "gets a moment", "keeps it together", "stays under control" unless unavoidable.
For recovery moments, vary the outcome language. Do not default to the same reset phrase every line.
Prefer specific consequence wording when appropriate, like edge-of-control or momentum loss.

Start with an action verb.
""".strip()


def build_prompt(
    spec: Dict[str, Any],
    state: NarrativeState,
    draft_line: str,
    recent_openers: List[str],
    style_examples: List[str],
) -> str:
    avoid = ", ".join([o for o in recent_openers[-14:] if o]) or "none"
    examples_block = (
        "\n".join([f"- {x}" for x in style_examples]) if style_examples else "- (none)"
    )
    forbidden = ", ".join(sorted(TIRED_PHRASES))
    recovery_style_block = (
        "Recovery language guidance:\n"
        "- Avoid repetitive reset endings like 'gathers it up', 'keeps it together', 'carries on', 'stays under control'.\n"
        "- Vary with consequence/evaluation phrasing when it fits the event.\n"
        "- Examples for cadence only: 'just about saves it', 'that was right on the edge', 'lucky to keep it straight', 'that costs momentum'.\n\n"
        if bool(spec.get("recovery")) or bool(spec.get("instability"))
        else ""
    )

    return (
        f"Window summary: {spec['sequence']} (events={spec['count']})\n"
        f"Dominant event: {spec['dominant_event']}\n"
        f"Dominant severity: {float(spec.get('dominant_severity', 0.0)):.2f}\n"
        f"Tone target: {spec.get('tone_level', 'calm')}\n"
        f"Cadence target: {'concise (1-2 clauses)' if int(spec.get('count', 0)) <= 2 else 'standard (2-3 clauses)'}\n"
        f"Headline metric (optional): {spec.get('headline') or 'none'}\n"
        f"Metric reason: {spec.get('metric_reason') or 'none'}\n"
        f"Intensity peak: {spec['intensity_peak']}\n"
        f"Hedge required: {spec['needs_hedge']}\n"
        f"Soft hedge: {spec['soft_hedge']}\n"
        f"Drama required: {spec['needs_drama']}\n"
        f"Avoid reusing these opening words: {avoid}\n"
        f"Avoid these tired phrases: {forbidden}\n\n"
        f"{recovery_style_block}"
        "Style examples (cadence only, do NOT copy):\n"
        f"{examples_block}\n\n"
        f"Draft line to rewrite:\n{draft_line}\n\n"
        "Task: Rewrite the draft into natural broadcast cadence while obeying every rule.\n"
        "Match wording energy to Tone target exactly.\n"
        "If Tone target is calm: keep language measured and avoid alarm words.\n"
        "If Tone target is urgent: make it sharper than calm, but no drama phrase.\n"
        "If Tone target is dramatic: use one strong impact phrase and heightened urgency.\n"
        "If Intensity peak is True, capitalize ONLY one short opening clause (2-8 words).\n"
        "Do NOT write the whole sentence in all caps.\n"
        "Do NOT include any meta/style tags or bracketed notes (for example '(Tone: calm)' or '[urgent]')."
    )


# =========================
# Postprocess + validation
# =========================


def _words(text: str) -> List[str]:
    return re.findall(r"\b[\w']+\b", text.lower())


def line_opener(text: str, n: int = 3) -> str:
    w = _words(text)
    return " ".join(w[:n]) if w else ""


def line_opener_key(text: str, n: int = 2) -> str:
    w = _words(text)
    if len(w) >= 2 and w[0] == "looks" and w[1] == "like":
        w = w[2:]
    elif len(w) >= 2 and w[0] == "might" and w[1] == "have":
        w = w[2:]
    if not w:
        return ""
    return " ".join(w[:n])


def postprocess_line(text: str) -> str:
    t = re.sub(r"\s{2,}", " ", text.strip())

    # Remove accidental meta/style tags from model output, e.g. "(Tone: calm)".
    t = re.sub(r"\s*[\(\[]\s*tone\s*:\s*[^\)\]]*[\)\]]\s*", " ", t, flags=re.IGNORECASE)

    # Normalize capitalization
    if t and t[0].islower():
        t = t[0].upper() + t[1:]

    # Kill the specific "semicolon + and it" disease
    t = re.sub(r";\s*and it\b", ", and", t, flags=re.IGNORECASE)

    # Ban semicolons completely: convert remaining to commas
    t = t.replace(";", ",")

    # Remove awkward “and it’s a clean acceleration” summary
    t = re.sub(
        r"\band it's a clean acceleration\b",
        "and it drives off cleanly",
        t,
        flags=re.IGNORECASE,
    )
    # Reduce 'and it' spam
    if t.lower().count("and it") > 1:
        parts = re.split(r"\band it\b", t, flags=re.IGNORECASE)
        t = parts[0] + "and it".join([parts[1]] + [" and ".join(parts[2:])])

    # Reduce pronoun chant if "it" shows up too much
    if len(re.findall(r"\bit\b", t.lower())) >= 3:
        t = re.sub(r"\bit\b", "the car", t, count=1, flags=re.IGNORECASE)

    # Cleanup spacing
    t = re.sub(r"\s+([,.])", r"\1", t)
    t = re.sub(r",\s*,", ",", t)

    # Normalize unit abbreviations to broadcast style.
    t = re.sub(r"\brpm\b", "RPM", t, flags=re.IGNORECASE)
    t = re.sub(r"\bkm/h\b", "km/h", t, flags=re.IGNORECASE)

    # Final spacing cleanup after all transforms.
    t = re.sub(r"\s{2,}", " ", t)
    t = re.sub(r"\s+([,.])", r"\1", t)
    return t.strip()


def validate_line(
    text: str,
    spec: Dict[str, Any],
    recent_lines: List[str],
    used_examples: List[str],
) -> Tuple[bool, str]:
    t = text.strip()
    tl = t.lower()

    if any(q in t for q in ['"', "“", "”", "’", "‘"]):
        return False, "contains quotes"
    if "!" in t:
        return False, "contains exclamation"
    if "-" in t or "—" in t:
        return False, "contains dash"
    if ";" in t:
        return False, "contains semicolon"
    if t.count(".") > 1:
        return False, "too many sentences"
    if re.search(r"\bthen and\b", tl) or re.search(r"\band and\b", tl):
        return False, "broken conjunction"
    if re.search(r"\bton(e|al)\s*:", tl):
        return False, "contains meta tone label"
    if re.search(r"\btone target\b", tl):
        return False, "contains prompt meta text"
    if re.search(r"[\(\[][^\)\]]{0,60}\b(tone|urgent|dramatic|calm)\b[^\)\]]*[\)\]]", tl):
        return False, "contains bracketed meta note"
    alpha = re.sub(r"[^A-Za-z]+", "", t)
    if len(alpha) >= 10 and t == t.upper():
        return False, "whole line caps not allowed"

    w = re.findall(r"\b[\w']+\b", t)
    if not (8 <= len(w) <= 22):
        return False, f"word count {len(w)}"
    if int(spec.get("count", 0)) <= 2 and t.count(",") > 1:
        return False, "too many clauses for concise cadence"

    nums = re.findall(r"\d+(?:\.\d+)?", t)
    headline = str(spec.get("headline") or "")
    has_metric = bool(spec.get("has_metric")) and bool(headline)
    if has_metric:
        if len(nums) != 1:
            return False, f"number count {len(nums)}"
        if headline.lower() not in tl:
            return False, "missing headline metric"
        if re.match(r"^\s*\d", t):
            return False, "starts with a number"
    else:
        if len(nums) != 0:
            return False, f"number count {len(nums)}"
        if "km/h" in tl or "rpm" in tl:
            return False, "unit used without metric"

    for term in BANNED_TERMS:
        if term in tl:
            return False, f"banned term: {term}"

    for phrase in BANNED_PURPLE:
        if phrase in tl:
            return False, f"purple prose: {phrase}"

    for phrase in TIRED_PHRASES:
        if phrase in tl and _recent_contains(recent_lines, phrase, n=10):
            return False, f"tired phrase repeated: {phrase}"

    if tl.count("and it") > 1:
        return False, "too many 'and it'"

    # Hedge rules
    needs_hedge = bool(spec["needs_hedge"])
    soft_hedge = bool(spec["soft_hedge"])
    has_looks = "looks like" in tl
    has_might = "might have" in tl
    hedge_count = int(has_looks) + int(has_might)
    if needs_hedge:
        if hedge_count != 1:
            return False, "hedge required but missing or multiple"
    else:
        if not soft_hedge and hedge_count > 0:
            return False, "hedge used when not allowed"

    # Drama rules
    needs_drama = bool(spec["needs_drama"])
    drama_hits = sum(1 for p in DRAMA_PHRASES if p in tl)
    if needs_drama:
        if drama_hits != 1:
            return False, "drama required but missing or multiple"
    else:
        if drama_hits > 0:
            return False, "drama used when not allowed"

    # Verb-led opener (skip hedge)
    opener_words = _words(t)
    if not opener_words:
        return False, "empty output"

    idx = 0
    if (
        len(opener_words) >= 2
        and opener_words[0] == "looks"
        and opener_words[1] == "like"
    ):
        idx = 2
    elif (
        len(opener_words) >= 2
        and opener_words[0] == "might"
        and opener_words[1] == "have"
    ):
        idx = 2

    if idx >= len(opener_words):
        return False, "bad opener"

    first = opener_words[idx]
    if first not in VERB_OPENERS:
        return False, f"opener not verb-led: {first}"

    # Anti repetition: opener
    opener3 = line_opener(t, 3)
    recent_openers = [line_opener(x, 3) for x in recent_lines[-8:]]
    if opener3 and opener3 in recent_openers:
        return False, "repeated opener"
    opener2 = line_opener_key(t, 2)
    recent_opener2 = [line_opener_key(x, 2) for x in recent_lines[-8:]]
    if opener2 and opener2 in recent_opener2:
        return False, "repeated opener root"

    # Anti repetition: cadence stamp (template smell)
    stamp = re.sub(r"\d+(?:\.\d+)?\s*(?:km/h|rpm)", "<NUM>", tl)
    stamp = re.sub(r"\b(the car|it)\b", "<SUBJ>", stamp)
    recent_stamps = [
        re.sub(r"\d+(?:\.\d+)?\s*(?:km/h|rpm)", "<NUM>", x.lower())
        for x in recent_lines[-4:]
    ]
    if stamp in recent_stamps:
        return False, "repeated cadence stamp"

    return True, "ok"


# =========================
# Generation
# =========================


def llm_rewrite(llm, prompt: str, temperature: float = 0.78) -> str:
    out = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_tokens=90,
        temperature=temperature,
        top_p=0.9,
        repeat_penalty=1.20,
        frequency_penalty=0.32,
        presence_penalty=0.20,
    )
    return out["choices"][0]["message"]["content"].strip()


def generate_line(
    llm,
    spec: Dict[str, Any],
    state: NarrativeState,
    recent_lines: List[str],
    recent_openers: List[str],
    style_examples: List[str],
    max_retries: int = 14,
) -> str:
    draft = make_beat_plan(spec, recent_lines)

    if llm is None:
        return postprocess_line(draft)

    prompt = build_prompt(spec, state, draft, recent_openers, style_examples)
    last_text = draft

    for attempt in range(max_retries):
        temp = 0.74 + 0.05 * min(attempt, 6)
        last_text = postprocess_line(llm_rewrite(llm, prompt, temperature=temp))

        ok, reason = validate_line(last_text, spec, recent_lines, style_examples)
        if ok:
            return last_text

        prompt = (
            build_prompt(spec, state, draft, recent_openers, style_examples)
            + f"\n\nYour last output was invalid: {reason}.\n"
            "Fix it. Spoken cadence. No semicolons. No dashes. Max one 'and it'.\n"
            "One sentence, 8–22 words. If cadence target is concise, keep 1–2 clauses only.\n"
            "If metric exists, one number only matching it; otherwise zero numbers. Start with an action verb.\n"
            "Do not add banned race context. Do not copy examples."
        )

    return postprocess_line(draft)


# =========================
# Pipeline
# =========================


def run_pipeline_stream(
    llm,
    csv_path: str = "events.csv",
    out_path: str = "commentary.csv",
    style_bank_path: str = "commentary_bank.csv",
    window_s: float = 4.0,
    max_gap_s: float = 1.5,
    seed: int = 7,
) -> Iterator[Dict[str, Any]]:
    random.seed(seed)

    events = read_events_csv(csv_path)
    bursts = group_events_by_window(events, window_s=window_s, max_gap_s=max_gap_s)
    total_lines = len(bursts)

    bank_real_path = (
        bundled_path(style_bank_path)
        if not Path(style_bank_path).exists()
        else style_bank_path
    )
    style_bank = read_style_bank_csv(bank_real_path)

    outputs: List[str] = []
    output_rows: List[Tuple[float, float, str]] = []
    openers: List[str] = []
    state = NarrativeState()
    last_end_s = -1e9
    last_emitted_event_ts = -1e9

    # Playback scheduler + cooldown to prevent overlapping/unnatural rapid calls.
    speaking_wps = 2.6
    min_gap_s = 0.5
    cooldown_window_s = 1.8
    cooldown_severity_threshold = 0.78

    for i, burst in tqdm(enumerate(bursts), total=total_lines):
        spec = burst_spec(burst)
        state = update_state(state, spec)

        event_ts = float(spec["timestamp_s"])
        dominant_event = str(spec["dominant_event"])
        dominant_severity = float(spec["dominant_severity"])
        high_priority = (
            dominant_event in HIGH_DRAMA_EVENTS
            or dominant_event in URGENT_EVENTS
            or dominant_severity >= 0.88
        )

        if (
            (event_ts - last_emitted_event_ts) < cooldown_window_s
            and not high_priority
            and dominant_severity < cooldown_severity_threshold
        ):
            yield {
                "stage": "text_generation",
                "status": "progress",
                "processed_lines": i + 1,
                "total_lines": total_lines,
                "generated_lines": len(outputs),
                "skipped": True,
            }
            continue

        seed_key = f"{seed}-{i}-{spec['dominant_event']}-{spec['headline']}"
        examples = sample_style_examples(
            style_bank, spec["dominant_event"], k=8, seed_key=seed_key
        )

        line = generate_line(
            llm=llm,
            spec=spec,
            state=state,
            recent_lines=outputs,
            recent_openers=openers,
            style_examples=examples,
        )
        line = emphasize_intensity_clause(line, spec)
        # Re-apply output normalization after optional emphasis uppercasing.
        line = postprocess_line(line)

        words = re.findall(r"\b[\w']+\b", line)
        duration_s = max(1.2, len(words) / speaking_wps)
        broadcast_ts = max(event_ts, last_end_s + min_gap_s)

        end_ts = broadcast_ts + duration_s

        outputs.append(line)
        output_rows.append((broadcast_ts, end_ts, line))
        openers.append(line_opener(line, 3))
        state = replace(state, last_opener=openers[-1])
        last_end_s = end_ts
        last_emitted_event_ts = event_ts
        yield {
            "stage": "text_generation",
            "status": "progress",
            "processed_lines": i + 1,
            "total_lines": total_lines,
            "generated_lines": len(outputs),
            "line": line,
            "timestamp": f"{broadcast_ts:.3f}",
            "skipped": False,
        }

    with Path(out_path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["start_time", "end_time", "text"])
        for start_time_s, end_time_s, line in output_rows:
            writer.writerow([round(start_time_s, 3), round(end_time_s, 3), line])

        print(f"Text commentary written to {out_path}")
    yield {
        "stage": "text_generation",
        "status": "done",
        "processed_lines": total_lines,
        "total_lines": total_lines,
        "generated_lines": len(outputs),
        "commentary_path": out_path,
        "outputs": outputs,
    }


def run_pipeline(
    llm,
    csv_path: str = "events.csv",
    out_path: str = "commentary.csv",
    style_bank_path: str = "commentary_bank.csv",
    window_s: float = 4.0,
    max_gap_s: float = 1.5,
    seed: int = 7,
) -> List[str]:
    final_outputs: List[str] = []
    for event in run_pipeline_stream(
        llm=llm,
        csv_path=csv_path,
        out_path=out_path,
        style_bank_path=style_bank_path,
        window_s=window_s,
        max_gap_s=max_gap_s,
        seed=seed,
    ):
        if event.get("status") == "done":
            final_outputs = list(event.get("outputs", []))
    return final_outputs


if __name__ == "__main__":
    model_path = bundled_path(
        os.path.join("models", "granite-3.3-2b-instruct-Q4_K_M.gguf")
    )
    llm = Llama(model_path=model_path, n_ctx=2048)

    run_pipeline(
        llm, "events.csv", "commentary.csv", style_bank_path="commentary_bank.csv"
    )
