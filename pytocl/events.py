
"""Code to detect events from telemetry samples."""
from __future__ import annotations

import argparse
import ast
import os
from enum import Enum
from typing import Iterable, List, Optional, Tuple

import pandas as pd


class Event(Enum):
    """Event names emitted by telemetry analysis."""

    LAP_START = "LapStart"
    LAP_COMPLETE = "LapComplete"
    HARD_BRAKING = "HardBraking"
    STRONG_ACCELERATION = "StrongAcceleration"
    LIFT_AND_COAST = "LiftAndCoast"
    UPSHIFT = "UpShift"
    DOWNSHIFT = "DownShift"
    OVER_REV = "OverRev"
    BOUNCING_LIMITER = "BouncingLimiter"
    CORNER_ENTRY = "CornerEntry"
    CORNER_EXIT = "CornerExit"
    TRACK_LIMITS_WARNING = "TrackLimitsWarning"
    OFF_TRACK = "OffTrack"
    RUNNING_WIDE = "RunningWide"
    SLIDE = "Slide"
    SPIN = "Spin"
    RECOVERY = "Recovery"
    CAR_AHEAD_CLOSE = "CarAheadClose"
    CAR_BEHIND_CLOSE = "CarBehindClose"
    OVERTAKE = "Overtake"
    BEING_OVERTAKEN = "BeingOvertaken"
    SIDE_BY_SIDE = "SideBySide"
    CONTACT = "Contact"
    DAMAGE_EVENT = "DamageEvent"
    STOPPED = "Stopped"
    STUCK = "Stuck"


def _coerce_numeric(df: pd.DataFrame, columns: Iterable[str]) -> None:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def _parse_tuple_cell(cell: object) -> Optional[Tuple[float, ...]]:
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return None
    if isinstance(cell, (list, tuple)):
        return tuple(float(x) for x in cell)
    if not isinstance(cell, str):
        return None
    text = cell.strip()
    if not text:
        return None
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return None
    if isinstance(parsed, (list, tuple)):
        return tuple(float(x) for x in parsed)
    return None


def _parse_tuple_column(df: pd.DataFrame, column: str) -> Optional[pd.Series]:
    if column not in df.columns:
        return None
    return df[column].apply(_parse_tuple_cell)


def _rising_edge(flag: pd.Series) -> pd.Series:
    return flag & ~flag.shift(1, fill_value=False)


def _falling_edge(flag: pd.Series) -> pd.Series:
    return ~flag & flag.shift(1, fill_value=False)


def _min_sector_distance(values: Optional[Tuple[float, ...]], indices: List[int]) -> float:
    if not values:
        return float("inf")
    size = len(values)
    clipped = [values[i] for i in indices if -size <= i < size]
    return min(clipped) if clipped else float("inf")


def _opponent_sector_ranges(length: int) -> Tuple[List[int], List[int], List[int]]:
    center = length // 2
    front = list(range(center - 2, center + 3))
    rear = list(range(0, 3)) + list(range(length - 3, length))
    side = [max(center - 9, 0), min(center + 9, length - 1)]
    side += [max(center - 10, 0), min(center + 10, length - 1)]
    side = sorted(set(side))
    return front, rear, side


def _emit_events(df: pd.DataFrame, mask: pd.Series, event: Event) -> List[dict]:
    rows = df.loc[mask, ["timestamp"]].copy()
    rows["event"] = event
    return rows.to_dict(orient="records")


def _event_type(event: Event) -> str:
    return event.name


def _base_severity(event: Event) -> float:
    return {
        Event.LAP_START: 0.2,
        Event.LAP_COMPLETE: 0.2,
        Event.HARD_BRAKING: 0.6,
        Event.STRONG_ACCELERATION: 0.5,
        Event.LIFT_AND_COAST: 0.3,
        Event.UPSHIFT: 0.2,
        Event.DOWNSHIFT: 0.2,
        Event.OVER_REV: 0.7,
        Event.BOUNCING_LIMITER: 0.85,
        Event.CORNER_ENTRY: 0.3,
        Event.CORNER_EXIT: 0.3,
        Event.TRACK_LIMITS_WARNING: 0.5,
        Event.OFF_TRACK: 0.9,
        Event.RUNNING_WIDE: 0.6,
        Event.SLIDE: 0.8,
        Event.SPIN: 1.0,
        Event.RECOVERY: 0.4,
        Event.CAR_AHEAD_CLOSE: 0.4,
        Event.CAR_BEHIND_CLOSE: 0.4,
        Event.OVERTAKE: 0.7,
        Event.BEING_OVERTAKEN: 0.7,
        Event.SIDE_BY_SIDE: 0.6,
        Event.CONTACT: 1.0,
        Event.DAMAGE_EVENT: 1.0,
        Event.STOPPED: 0.5,
        Event.STUCK: 0.9,
    }.get(event, 0.5)


def _event_priority(event: Event) -> int:
    return {
        Event.CONTACT: 100,
        Event.DAMAGE_EVENT: 100,
        Event.SPIN: 95,
        Event.OFF_TRACK: 90,
        Event.STUCK: 85,
        Event.SLIDE: 80,
        Event.RUNNING_WIDE: 70,
        Event.TRACK_LIMITS_WARNING: 65,
        Event.HARD_BRAKING: 60,
        Event.STRONG_ACCELERATION: 55,
        Event.OVERTAKE: 55,
        Event.BEING_OVERTAKEN: 55,
        Event.SIDE_BY_SIDE: 50,
        Event.CAR_AHEAD_CLOSE: 45,
        Event.CAR_BEHIND_CLOSE: 45,
        Event.OVER_REV: 40,
        Event.BOUNCING_LIMITER: 40,
        Event.STOPPED: 40,
        Event.RECOVERY: 35,
        Event.CORNER_ENTRY: 30,
        Event.CORNER_EXIT: 30,
        Event.LIFT_AND_COAST: 25,
        Event.UPSHIFT: 20,
        Event.DOWNSHIFT: 20,
        Event.LAP_START: 10,
        Event.LAP_COMPLETE: 10,
    }.get(event, 10)


def _bounded(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _event_confidence(score: float) -> float:
    return _bounded(score)


def _filter_events(
    events: List[dict],
    min_interval_s: float,
    default_cooldown_s: float,
) -> List[dict]:
    if not events:
        return []
    ordered = sorted(
        events,
        key=lambda item: (item["timestamp"], -_event_priority(item["event"])),
    )
    kept: List[dict] = []
    last_kept_time: Optional[float] = None
    last_kept_priority: Optional[int] = None
    last_event_time: dict[Event, float] = {}
    for item in ordered:
        timestamp = item["timestamp"]
        event = item["event"]
        if pd.isna(timestamp):
            continue
        prev_time = last_event_time.get(event)
        if prev_time is not None and (timestamp - prev_time) < default_cooldown_s:
            continue
        priority = _event_priority(event)
        if last_kept_time is not None and (timestamp - last_kept_time) < min_interval_s:
            if last_kept_priority is not None and priority > last_kept_priority:
                kept.pop()
                kept.append(item)
                last_kept_time = timestamp
                last_kept_priority = priority
                last_event_time[event] = timestamp
            continue
        kept.append(item)
        last_kept_time = timestamp
        last_kept_priority = priority
        last_event_time[event] = timestamp
    return kept


def detect_events(
    df: pd.DataFrame,
    session_id: str,
    car_id: str = "ego",
    min_event_interval_s: float = 0.25,
    event_cooldown_s: float = 0.5,
) -> pd.DataFrame:
    """Detect telemetry events and return a timestamped event DataFrame."""
    df = df.copy()
    numeric_columns = [
        "timestamp",
        "current_lap",
        "current_lap_time",
        "damage",
        "distance_raced",
        "distance_from_center",
        "fuel",
        "gear",
        "race_position",
        "rpm",
        "speed_x",
        "speed_y",
        "speed_z",
    ]
    _coerce_numeric(df, numeric_columns)

    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    dt = df["timestamp"].diff().fillna(0).clip(lower=0.0)

    if {"speed_x", "speed_y", "speed_z"}.issubset(df.columns):
        df["speed"] = (df["speed_x"] ** 2 + df["speed_y"] ** 2 + df["speed_z"] ** 2) ** 0.5
    elif "speed_x" in df.columns:
        df["speed"] = df["speed_x"].abs()
    else:
        df["speed"] = 0.0

    df["longitudinal_accel"] = df["speed_x"].diff().fillna(0) / dt.replace(0, 1)
    rpm_delta = df["rpm"].diff().fillna(0)

    if "distance_from_start" in df.columns:
        df["dist_from_start_m"] = pd.to_numeric(df["distance_from_start"], errors="coerce")
    elif "distance_raced" in df.columns:
        df["dist_from_start_m"] = pd.to_numeric(df["distance_raced"], errors="coerce")
    else:
        df["dist_from_start_m"] = pd.NA

    if "distance_from_center" in df.columns:
        df["track_pos"] = pd.to_numeric(df["distance_from_center"], errors="coerce")
    else:
        df["track_pos"] = pd.NA

    if "current_lap" in df.columns:
        df["lap"] = pd.to_numeric(df["current_lap"], errors="coerce").astype("Int64")
    else:
        df["lap"] = pd.NA

    if "dist_from_start_m" in df.columns and df["dist_from_start_m"].notna().any():
        track_length = df["dist_from_start_m"].max()
        sector_length = track_length / 3.0 if track_length and track_length > 0 else None
        if sector_length:
            sector_float = (df["dist_from_start_m"] / sector_length).clip(lower=0)
            sector_index = sector_float.apply(
                lambda value: int(value) if pd.notna(value) else pd.NA
            )
            df["sector"] = sector_index.astype("Int64") + 1
            df.loc[df["sector"] > 3, "sector"] = 3
        else:
            df["sector"] = pd.NA
    else:
        df["sector"] = pd.NA

    events: List[dict] = []

    if "current_lap" in df.columns:
        lap_change = df["current_lap"].diff().fillna(0) > 0
        lap_start = lap_change | (df.index == 0)
        events += _emit_events(df, lap_start, Event.LAP_START)
        lap_complete_rows = df.index[lap_change].to_series().apply(lambda idx: idx - 1)
        lap_complete_rows = lap_complete_rows[lap_complete_rows >= 0]
        events += _emit_events(df, df.index.isin(lap_complete_rows), Event.LAP_COMPLETE)

    hard_braking = df["longitudinal_accel"] <= -5.0
    events += _emit_events(df, _rising_edge(hard_braking), Event.HARD_BRAKING)

    strong_accel = df["longitudinal_accel"] >= 3.0
    events += _emit_events(df, _rising_edge(strong_accel), Event.STRONG_ACCELERATION)

    lift_and_coast = df["longitudinal_accel"].between(-3.0, -1.0) & (rpm_delta < -50)
    events += _emit_events(df, _rising_edge(lift_and_coast), Event.LIFT_AND_COAST)

    if "gear" in df.columns:
        gear_delta = df["gear"].diff().fillna(0)
        events += _emit_events(df, gear_delta > 0, Event.UPSHIFT)
        events += _emit_events(df, gear_delta < 0, Event.DOWNSHIFT)

    if "rpm" in df.columns and df["rpm"].notna().any():
        rpm_limit = df["rpm"].quantile(0.98)
        over_rev = df["rpm"] >= rpm_limit
        events += _emit_events(df, _rising_edge(over_rev), Event.OVER_REV)
        bouncing = over_rev & (df["longitudinal_accel"] <= 0.2)
        events += _emit_events(df, _rising_edge(bouncing), Event.BOUNCING_LIMITER)

    if "angle" in df.columns:
        df["angle"] = pd.to_numeric(df["angle"], errors="coerce")
        cornering = df["angle"].abs() >= 0.1
        events += _emit_events(df, _rising_edge(cornering), Event.CORNER_ENTRY)
        events += _emit_events(df, _falling_edge(cornering), Event.CORNER_EXIT)

    if "distance_from_center" in df.columns:
        df["distance_from_center"] = pd.to_numeric(df["distance_from_center"], errors="coerce")
        off_track = df["distance_from_center"].abs() > 1.0
        warning = df["distance_from_center"].abs().between(0.9, 1.0, inclusive="left")
        running_wide = (
            df["distance_from_center"].abs() > 0.9
        ) & (df["distance_from_center"].abs().diff().fillna(0) > 0)
        events += _emit_events(df, _rising_edge(warning), Event.TRACK_LIMITS_WARNING)
        events += _emit_events(df, _rising_edge(off_track), Event.OFF_TRACK)
        events += _emit_events(df, _rising_edge(running_wide), Event.RUNNING_WIDE)
    else:
        off_track = pd.Series(False, index=df.index)

    slide = (df["speed_y"].abs() >= 3.0) & (df["speed"] >= 5.0) if "speed_y" in df.columns else pd.Series(False, index=df.index)
    events += _emit_events(df, _rising_edge(slide), Event.SLIDE)

    spin = (df["angle"].abs() >= 1.2) & (df["speed"] >= 2.0) if "angle" in df.columns else pd.Series(False, index=df.index)
    events += _emit_events(df, _rising_edge(spin), Event.SPIN)

    recovery = _falling_edge(off_track | slide | spin)
    events += _emit_events(df, recovery, Event.RECOVERY)

    opponents = _parse_tuple_column(df, "opponents")
    if opponents is not None and opponents.notna().any():
        sample = opponents.dropna().iloc[0]
        if sample is None:
            sample = ()
        if sample:
            front_idx, rear_idx, side_idx = _opponent_sector_ranges(len(sample))
            front_min = opponents.apply(lambda vals: _min_sector_distance(vals, front_idx))
            rear_min = opponents.apply(lambda vals: _min_sector_distance(vals, rear_idx))
            side_min = opponents.apply(lambda vals: _min_sector_distance(vals, side_idx))
            car_ahead_close = front_min < 10.0
            car_behind_close = rear_min < 10.0
            side_by_side = side_min < 5.0
            events += _emit_events(df, _rising_edge(car_ahead_close), Event.CAR_AHEAD_CLOSE)
            events += _emit_events(df, _rising_edge(car_behind_close), Event.CAR_BEHIND_CLOSE)
            events += _emit_events(df, _rising_edge(side_by_side), Event.SIDE_BY_SIDE)

    if "race_position" in df.columns:
        pos_delta = df["race_position"].diff().fillna(0)
        events += _emit_events(df, pos_delta < 0, Event.OVERTAKE)
        events += _emit_events(df, pos_delta > 0, Event.BEING_OVERTAKEN)

    if "damage" in df.columns:
        damage_delta = df["damage"].diff().fillna(0)
        damage_event = damage_delta > 0
        events += _emit_events(df, damage_event, Event.CONTACT)
        events += _emit_events(df, damage_event, Event.DAMAGE_EVENT)

    stopped = df["speed"] < 0.5
    events += _emit_events(df, _rising_edge(stopped), Event.STOPPED)

    stopped_time = (dt.where(stopped, 0.0)).groupby((~stopped).cumsum()).cumsum()
    stuck = stopped & (stopped_time >= 3.0)
    events += _emit_events(df, _rising_edge(stuck), Event.STUCK)

    events = _filter_events(events, min_event_interval_s, event_cooldown_s)
    if not events:
        return pd.DataFrame(
            columns=[
                "event_id",
                "session_id",
                "car_id",
                "timestamp_s",
                "lap",
                "sector",
                "dist_from_start_m",
                "track_pos",
                "event_type",
                "severity",
                "confidence",
                "speed_mps",
                "longitudinal_accel_mps2",
                "rpm",
                "gear",
                "angle_rad",
                "lateral_speed_mps",
            ]
        )

    events_df = pd.DataFrame(events).dropna(subset=["timestamp"]).sort_values("timestamp")
    enriched = df.loc[events_df.index].copy()
    events_df = events_df.reset_index(drop=True)
    enriched = enriched.reset_index(drop=True)

    severity = []
    confidence = []
    for event, accel, angle, off_track_flag in zip(
        events_df["event"],
        enriched.get("longitudinal_accel", pd.Series(0, index=enriched.index)),
        enriched.get("angle", pd.Series(0, index=enriched.index)),
        off_track if isinstance(off_track, pd.Series) else pd.Series(False, index=enriched.index),
    ):
        base = _base_severity(event)
        if event in (Event.HARD_BRAKING, Event.STRONG_ACCELERATION):
            magnitude = abs(float(accel)) if pd.notna(accel) else 0.0
            sev = _bounded(base + (magnitude / 10.0))
            conf = _event_confidence(min(1.0, magnitude / 6.0))
        elif event in (Event.SPIN, Event.SLIDE):
            mag = abs(float(angle)) if pd.notna(angle) else 0.0
            sev = _bounded(base + mag / 2.0)
            conf = _event_confidence(min(1.0, mag / 1.5))
        elif event in (Event.OFF_TRACK, Event.TRACK_LIMITS_WARNING):
            sev = _bounded(base + (0.2 if off_track_flag else 0.0))
            conf = _event_confidence(0.8 if off_track_flag else 0.6)
        else:
            sev = base
            conf = 0.7
        severity.append(sev)
        confidence.append(conf)

    result = pd.DataFrame(
        {
            "event_id": range(1, len(events_df) + 1),
            "session_id": session_id,
            "car_id": car_id,
            "timestamp_s": events_df["timestamp"],
            "lap": enriched.get("lap"),
            "sector": enriched.get("sector"),
            "dist_from_start_m": enriched.get("dist_from_start_m"),
            "track_pos": enriched.get("track_pos"),
            "event_type": events_df["event"].apply(_event_type),
            "severity": severity,
            "confidence": confidence,
            "speed_mps": enriched.get("speed"),
            "longitudinal_accel_mps2": enriched.get("longitudinal_accel"),
            "rpm": enriched.get("rpm"),
            "gear": enriched.get("gear"),
            "angle_rad": enriched.get("angle"),
            "lateral_speed_mps": enriched.get("speed_y"),
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Telemetry event detection.")
    parser.add_argument(
        "--input_path",
        help="Path to telemetry_samples.csv.",
        default="telemetry_samples.csv",
        type=str,
    )
    parser.add_argument(
        "--output_path",
        help="Output path to events.csv.",
        default="events.csv",
        type=str,
    )
    parser.add_argument(
        "--session_id",
        help="Session identifier for stitching multiple runs.",
        default="session-1",
        type=str,
    )
    parser.add_argument("--car_id", help="Car identifier.", default="ego", type=str)
    parser.add_argument(
        "--min_event_interval_s",
        help="Minimum time between any two recorded events.",
        default=0.25,
        type=float,
    )
    parser.add_argument(
        "--event_cooldown_s",
        help="Minimum time between two events of the same type.",
        default=0.5,
        type=float,
    )
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path or "events.csv"

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    events_df = detect_events(
        df,
        session_id=args.session_id,
        car_id=args.car_id,
        min_event_interval_s=args.min_event_interval_s,
        event_cooldown_s=args.event_cooldown_s,
    )
    events_df.to_csv(output_path, index=False)
    print(f"Wrote {len(events_df)} events to {output_path}")


if __name__ == "__main__":
    main()
