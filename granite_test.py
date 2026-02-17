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

# # --- Aggregated event burst pipeline (below line 106) ---

import csv
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
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


def read_events_csv(path: str) -> List[Event]:
    events: List[Event] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            events.append(
                Event(
                    event_type=row["event_type"],
                    severity=float(row["severity"]),
                    confidence=float(row["confidence"]),
                    timestamp_s=float(row["timestamp_s"]),
                    lap=int(float(row["lap"])),
                    sector=int(float(row["sector"])),
                    dist_from_start_m=float(row["dist_from_start_m"]),
                    track_pos=float(row["track_pos"]),
                    speed_mps=float(row["speed_mps"]),
                    rpm=float(row["rpm"]),
                    gear=int(float(row["gear"])),
                    longitudinal_accel_mps2=float(
                        row["longitudinal_accel_mps2"] or 0.0
                    ),
                )
            )
    return events


def pick_headline_metric(ev: Event) -> str:
    # You choose the metric; don't let the model "pick".
    speed_kmh = ev.speed_mps * 3.6
    if ev.event_type in ("SPIN", "OFFTRACK", "LOCKUP"):
        return f"speed_kmh={speed_kmh:.0f}"
    return f"rpm={int(round(ev.rpm/100)*100)}"


def format_event_compact(ev: Event) -> str:
    return (
        f"type={ev.event_type} sev={ev.severity:.2f} conf={ev.confidence:.2f} "
        f"{pick_headline_metric(ev)} gear={ev.gear}"
    )


def group_events_by_time(
    events: List[Event], max_gap_s: float = 5.0, max_events_per_burst: int = 3
) -> List[List[Event]]:
    if not events:
        return []
    events_sorted = sorted(events, key=lambda e: e.timestamp_s)
    bursts: List[List[Event]] = []
    current: List[Event] = [events_sorted[0]]

    for ev in events_sorted[1:]:
        gap = ev.timestamp_s - current[-1].timestamp_s
        if gap <= max_gap_s and len(current) < max_events_per_burst:
            current.append(ev)
        else:
            bursts.append(current)
            current = [ev]
    bursts.append(current)
    return bursts


def summarize_burst_window(burst: List[Event]) -> Tuple[str, Dict[str, float]]:
    times = [e.timestamp_s for e in burst]
    laps = [e.lap for e in burst]
    sectors = [e.sector for e in burst]
    dists = [e.dist_from_start_m for e in burst]
    tpos = [e.track_pos for e in burst]
    return (
        f"Burst window: time {min(times):.2f} to {max(times):.2f} "
        f"(dt {max(times)-min(times):.2f}), "
        f"lap {min(laps)}, sector {min(sectors)}, "
        f"distance {min(dists):.2f} to {max(dists):.2f}, "
        f"track_pos {min(tpos):.2f}-{max(tpos):.2f}.",
        {"t_start": min(times), "t_end": max(times)},
    )


def format_event_line(ev: Event) -> str:
    speed_kmh = ev.speed_mps * 3.6
    # Reduce numeric precision to feel more like human commentary.
    rpm_rounded = int(round(ev.rpm / 100.0) * 100)
    speed_rounded = round(speed_kmh, 1)
    time_rounded = round(ev.timestamp_s, 1)
    return (
        f"{ev.event_type} severity={ev.severity:.2f} confidence={ev.confidence:.2f} "
        f"at time={time_rounded:.1f}, speed={speed_rounded:.1f}, "
        f"rpm={rpm_rounded}, gear={ev.gear}, accel={ev.longitudinal_accel_mps2:.2f}"
    )


def build_burst_prompt(burst: List[Event], burst_index: int) -> str:
    # Provide a varied style palette to reduce repetitive openings.
    moods = [
        "tense and clinical",
        "urgent and punchy",
        "measured then explosive",
        "calm with a sharp twist",
        "tight and breathless",
        "focused and precise",
    ]
    verbs = [
        "clips",
        "wriggles",
        "hooks",
        "snaps",
        "clatters",
        "stabs",
        "surges",
        "lunges",
        "scrubs",
        "skips",
    ]
    forbidden_openers = [
        "Breath-held moment",
        "Sudden jolt",
        "Split-second scare",
        "Calm then chaos",
        "Momentum shift",
        "Massive",
        "Big",
        "Major",
    ]
    header, _ = summarize_burst_window(burst)
    # Sort by severity (desc), tie-break by timestamp for stable ordering.
    ordered = sorted(burst, key=lambda e: (-e.severity, e.timestamp_s))
    lines = [format_event_line(ev) for ev in ordered]
    # Provide explicit style constraints to avoid generic repetition and name invention.
    style = (
        f"Mood: {moods[burst_index % len(moods)]}. "
        f"Verb bank: {', '.join(verbs)}. "
        "Do not start with any forbidden opener. "
        "Avoid repeating any opener used in the previous output. "
        "Never mention driver names or teams; use 'the car' only. "
        "Use one concrete metric in the first 8 words."
        f"Forbidden openers: {', '.join(forbidden_openers)}."
    )
    return (
        f"{header}\n"
        "Events in order (highest severity first):\n" + "\n".join(lines)
        # + "\nStyle: "
        # + style
        + "\nTask: Produce commentary."
    )


events = read_events_csv("events.csv")
bursts = group_events_by_time(events, max_gap_s=8.0, max_events_per_burst=10)
outputs = []

system = """
You are an F1-style live race commentator with David Croft energy: short, sharp, punchy.

Output ONE sentence only, 8-14 words, present tense, staccato rhythm.
Use fragments and quick clauses. No filler. No hedging unless required below.
Max TWO clauses. Avoid commas; use periods or semicolons only.
Use exactly ONE number only. Prefer rounded numbers (e.g., 60 km/h, 5000 rpm, 15 seconds).
If event.confidence < 0.60: hedge with "looks like" or "might have".
If event.severity >= 0.70: use ONE dramatic phrase: "big moment", "massive scare", or "heavy hit".
NEVER invent: positions, overtakes, penalties, lap record, pit crew, fans, the field, the pack, "reclaiming the lead".
No driver/team names. If a name appears, replace with "the car".

Prefer F1 vocabulary, some examples are the following, use sparingly:
"late on the brakes", "locks up", "snaps oversteer", "rides the kerb", "cuts the apex",
"gets it rotated", "traction on exit", "wriggle", "tidy recovery", "keeps it pointing straight".

Example style (do NOT copy): "Late on the brakes. Very late. He's in trouble here."
"""
# BANNED words/phrases:
# "shockwaves", "clatter", "speedster", "trajectory", "plunges", "looms", "scrambling".

# End the sentence with " <END>"

for i, burst in enumerate(bursts, start=1):
    # Use burst index to rotate openers across prompts, not lap number.
    prompt = build_burst_prompt(burst, i)
    print(f"\n--- Burst {i} Prompt ---\n{prompt}\n")

    out = llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": system,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        max_tokens=100,
        temperature=0.9,
        top_p=0.9,
        stop=["<END>"],
        repeat_penalty=1.15,
        frequency_penalty=0.3,
        presence_penalty=0.1,
    )

    result = out["choices"][0]["message"]["content"].strip()
    outputs.append(result)
    # print(f"result {i} = '{result}'")

print("\n".join(outputs), file=open("commentary.txt", "w"))

# --- Aggregated event burst pipeline (updated) ---

# import csv
# import re
# from dataclasses import dataclass
# from typing import Dict, List, Tuple, Optional


# END_TOKEN = "<END>"


# @dataclass
# class Event:
#     event_type: str
#     severity: float
#     confidence: float
#     timestamp_s: float
#     lap: int
#     sector: int
#     dist_from_start_m: float
#     track_pos: float
#     speed_mps: float
#     rpm: float
#     gear: int
#     longitudinal_accel_mps2: float


# def read_events_csv(path: str) -> List[Event]:
#     events: List[Event] = []
#     with open(path, newline="") as f:
#         reader = csv.DictReader(f)
#         for row in reader:
#             events.append(
#                 Event(
#                     event_type=row["event_type"].strip(),
#                     severity=float(row["severity"]),
#                     confidence=float(row["confidence"]),
#                     timestamp_s=float(row["timestamp_s"]),
#                     lap=int(float(row["lap"])),
#                     sector=int(float(row["sector"])),
#                     dist_from_start_m=float(row["dist_from_start_m"]),
#                     track_pos=float(row["track_pos"]),
#                     speed_mps=float(row["speed_mps"]),
#                     rpm=float(row["rpm"]),
#                     gear=int(float(row["gear"])),
#                     longitudinal_accel_mps2=float(
#                         row.get("longitudinal_accel_mps2") or 0.0
#                     ),
#                 )
#             )
#     return events


# def group_events_by_time(
#     events: List[Event], max_gap_s: float = 5.0, max_events_per_burst: int = 3
# ) -> List[List[Event]]:
#     """
#     Groups by time using gap between consecutive events.
#     Keeps bursts small; we will later compress further anyway.
#     """
#     if not events:
#         return []
#     events_sorted = sorted(events, key=lambda e: e.timestamp_s)
#     bursts: List[List[Event]] = []
#     current: List[Event] = [events_sorted[0]]

#     for ev in events_sorted[1:]:
#         gap = ev.timestamp_s - current[-1].timestamp_s
#         if gap <= max_gap_s and len(current) < max_events_per_burst:
#             current.append(ev)
#         else:
#             bursts.append(current)
#             current = [ev]
#     bursts.append(current)
#     return bursts


# def summarize_burst_window(burst: List[Event]) -> Tuple[str, Dict[str, float]]:
#     times = [e.timestamp_s for e in burst]
#     laps = [e.lap for e in burst]
#     sectors = [e.sector for e in burst]
#     dists = [e.dist_from_start_m for e in burst]
#     tpos = [e.track_pos for e in burst]

#     t_start, t_end = min(times), max(times)
#     return (
#         f"WINDOW time_s={t_start:.2f}->{t_end:.2f} dt={t_end - t_start:.2f}; "
#         f"lap={min(laps)} sector={min(sectors)}; "
#         f"dist_m={min(dists):.0f}->{max(dists):.0f}; "
#         f"track_pos={min(tpos):.2f}->{max(tpos):.2f}",
#         {"t_start": t_start, "t_end": t_end},
#     )


# def speed_kmh(ev: Event) -> float:
#     return ev.speed_mps * 3.6


# def rpm_rounded(ev: Event) -> int:
#     return int(round(ev.rpm / 100.0) * 100)


# def choose_headline_metric(ev: Event) -> str:
#     """
#     Pick ONE headline metric in code so the model doesn't number-vomit.
#     """
#     et = ev.event_type.upper()

#     # For dramatic events, speed reads better in broadcast
#     if et in {"SPIN", "OFFTRACK", "LOCKUP", "CRASH", "CONTACT"}:
#         return f"speed_kmh={speed_kmh(ev):.0f}"

#     # If it's acceleration/shift-ish, RPM tends to sound right
#     if et in {"UPSHIFT", "DOWNSHIFT", "STRONGACCELERATION", "ACCEL", "ENGINE_SPIKE"}:
#         return f"rpm={rpm_rounded(ev)}"

#     # Otherwise default to speed
#     return f"speed_kmh={speed_kmh(ev):.0f}"


# def format_event_compact(ev: Event) -> str:
#     """
#     Compact, structured. No telemetry dump.
#     """
#     metric = choose_headline_metric(ev)
#     time_rounded = round(ev.timestamp_s, 1)
#     # Track position only matters if it's clearly off-line
#     tpos = f"{ev.track_pos:.2f}"
#     return (
#         f"type={ev.event_type} sev={ev.severity:.2f} conf={ev.confidence:.2f} "
#         f"time_s={time_rounded:.1f} {metric} gear={ev.gear} track_pos={tpos}"
#     )


# def select_key_events(burst: List[Event], k: int = 3) -> List[Event]:
#     """
#     Keep only the top-k “tellable” events.
#     """
#     ordered = sorted(burst, key=lambda e: (-e.severity, -e.confidence, e.timestamp_s))
#     return ordered[:k]


# def build_burst_prompt(burst: List[Event], burst_index: int) -> str:
#     """
#     Structured prompt: HEADLINE + SECONDARY + CONTEXT.
#     No rotating moods/verb banks. Those were pushing the model into cringe.
#     """
#     header, _ = summarize_burst_window(burst)

#     key = select_key_events(burst, k=3)
#     headline = key[0]
#     secondary = key[1:] if len(key) > 1 else []

#     headline_line = format_event_compact(headline)
#     secondary_lines = [format_event_compact(ev) for ev in secondary]

#     # Give the model one job: narrate the headline, optionally mention recovery from secondary.
#     prompt = [
#         header,
#         f"HEADLINE_EVENT {headline_line}",
#     ]
#     if secondary_lines:
#         prompt.append("SECONDARY_EVENTS")
#         prompt.extend(f"- {ln}" for ln in secondary_lines)

#     # Minimal guidance that doesn't contradict the system prompt
#     prompt.append(
#         "RULES: Focus on HEADLINE_EVENT first; optionally mention the recovery implied by SECONDARY_EVENTS. "
#         "Do not list multiple numbers. Use the headline metric already provided."
#     )
#     return "\n".join(prompt)


# def clean_output(text: str) -> str:
#     """
#     Strip END token and surrounding whitespace.
#     """
#     return text.replace(END_TOKEN, "").strip()


# def is_valid_commentary(text: str) -> bool:
#     """
#     Enforce constraints in code because the model will occasionally freestyle.
#     """
#     t = clean_output(text)

#     # Must be one sentence-ish: allow one terminal punctuation.
#     # This isn't perfect, but it kills most run-on disasters.
#     if t.count(".") > 1 or t.count("!") > 1 or t.count("?") > 1:
#         return False

#     words = t.split()
#     if not (10 <= len(words) <= 22):
#         return False

#     # Must contain at least one digit
#     if not re.search(r"\d", t):
#         return False

#     # Avoid some known hallucination magnets, even if the system says so
#     banned = ["lead", "reclaiming", "pit crew", "the field", "the pack", "fans"]
#     lower = t.lower()
#     if any(b in lower for b in banned):
#         return False

#     return True


# # -------------------------
# # System prompt (cleaned)
# # -------------------------
# system = f"""
# You are an F1 live commentator in a David Croft-like style: fast, punchy, vivid, but factual.

# Write ONE sentence, 12-20 words. Present tense. No quotes.
# Include exactly ONE number with unit (km/h or RPM).
# If confidence < 0.60: add a light hedge once ("seems", "looked like") but do NOT start the sentence with it.
# If severity >= 0.70: include exactly one of: "big moment", "massive scare", "heavy hit".
# Never mention: leader, positions, overtakes, penalties, pit crew, the pack/field, fans.
# No driver/team names: always "the car".
# End with <END>.

# Examples:
# Input: headline=LOCKUP; consequence=runs wide; recovery=recovers; number=198 km/h; sev=0.72; conf=0.66
# Output: Late on the brakes at 198 km/h, big moment, it locks up and skips wide, then regains control. <END>

# Input: headline=SPIN; consequence=rotation; recovery=catches it; number=142 km/h; sev=0.85; conf=0.80
# Output: Massive scare at 142 km/h, the car loops it in a snap, but gathers it up before the gravel. <END>

# Input: headline=OFFTRACK; consequence=wheel off; recovery=back on line; number=121 km/h; sev=0.55; conf=0.52
# Output: The car, at 121 km/h, seems to dip a wheel off, but it tucks back in and carries on. <END>
# """.strip()


# # -------------------------
# # Run bursts
# # -------------------------
# events = read_events_csv("events.csv")

# # Keep time grouping generous; we compress key events anyway.
# bursts = group_events_by_time(events, max_gap_s=8.0, max_events_per_burst=5)

# outputs: List[str] = []

# for i, burst in enumerate(bursts, start=1):
#     prompt = build_burst_prompt(burst, i)
#     print(f"\n--- Burst {i} Prompt ---\n{prompt}\n")

#     # First attempt: normal creative
#     out = llm.create_chat_completion(
#         messages=[
#             {"role": "system", "content": system},
#             {"role": "user", "content": prompt},
#         ],
#         max_tokens=60,
#         temperature=0.65,
#         top_p=0.9,
#         stop=[END_TOKEN],
#         repeat_penalty=1.15,
#         frequency_penalty=0.3,
#         presence_penalty=0.1,
#     )
#     result = out["choices"][0]["message"]["content"].strip()

#     # # Validate and retry once if needed
#     # if not is_valid_commentary(result):
#     #     repair_prompt = (
#     #         prompt
#     #         + "\nREPAIR: Your last output violated format. Output ONE sentence, 10-22 words, include exactly one number, end with "
#     #         + END_TOKEN
#     #     )
#     #     out2 = llm.create_chat_completion(
#     #         messages=[
#     #             {"role": "system", "content": system},
#     #             {"role": "user", "content": repair_prompt},
#     #         ],
#     #         max_tokens=60,
#     #         temperature=0.30,
#     #         top_p=0.9,
#     #         stop=[END_TOKEN],
#     #         repeat_penalty=1.2,
#     #         frequency_penalty=0.35,
#     #         presence_penalty=0.05,
#     #     )
#     #     result2 = out2["choices"][0]["message"]["content"].strip()
#     #     result = result2 if is_valid_commentary(result2) else result

#     outputs.append(clean_output(result))

# with open("commentary.txt", "w") as f:
#     f.write("\n".join(outputs))
