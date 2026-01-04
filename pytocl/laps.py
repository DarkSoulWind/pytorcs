"""Code to generate lap summaries."""
import argparse
import os
import pandas as pd

def summarize_laps(df: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = [
        "timestamp",
        "current_lap",
        "current_lap_time",
        "distance_raced",
        "fuel",
        "damage",
        "rpm",
        "speed_x",
        "speed_y",
        "speed_z",
        "distance_from_center",
    ]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["current_lap", "timestamp"]).copy()
    df = df.sort_values(["current_lap", "timestamp"])
    grouped = df.groupby("current_lap", sort=True)

    if "distance_from_center" in df.columns:
        df["offtrack"] = df["distance_from_center"].abs() > 1.0

    agg_map = {
        "start_timestamp": ("timestamp", "first"),
        "end_timestamp": ("timestamp", "last"),
        "samples": ("timestamp", "count"),
    }
    if "distance_raced" in df.columns:
        agg_map.update(
            {
                "start_distance_raced": ("distance_raced", "first"),
                "end_distance_raced": ("distance_raced", "last"),
            }
        )
    if "fuel" in df.columns:
        agg_map.update(
            {
                "start_fuel": ("fuel", "first"),
                "end_fuel": ("fuel", "last"),
            }
        )
    if "damage" in df.columns:
        agg_map.update(
            {
                "start_damage": ("damage", "first"),
                "end_damage": ("damage", "last"),
            }
        )
    if "speed_x" in df.columns:
        agg_map.update(
            {
                "avg_speed_x": ("speed_x", "mean"),
                "max_speed_x": ("speed_x", "max"),
            }
        )
    if "rpm" in df.columns:
        agg_map.update(
            {
                "avg_rpm": ("rpm", "mean"),
                "max_rpm": ("rpm", "max"),
            }
        )

    summary = grouped.agg(**agg_map).reset_index()
    summary["lap_time"] = summary["end_timestamp"] - summary["start_timestamp"]

    if "offtrack" in df.columns:
        offtrack_events = grouped["offtrack"].apply(
            lambda series: (series & ~series.shift(1, fill_value=False)).sum()
        )
        summary["offtrack_count"] = summary["current_lap"].map(offtrack_events)

    if "start_distance_raced" in summary.columns and "end_distance_raced" in summary.columns:
        summary["distance_raced_delta"] = (
            summary["end_distance_raced"] - summary["start_distance_raced"]
        )
    if "start_fuel" in summary.columns and "end_fuel" in summary.columns:
        summary["fuel_used"] = summary["start_fuel"] - summary["end_fuel"]
    if "start_damage" in summary.columns and "end_damage" in summary.columns:
        summary["damage_delta"] = summary["end_damage"] - summary["start_damage"]

    return summary

def main():
    parser = argparse.ArgumentParser(description="Lap summary generator from telemetry samples.")
    parser.add_argument(
        "--input_path",
        help="Path to telemetry_samples.csv.",
        default="telemetry_samples.csv",
        type=str,
    )
    parser.add_argument("--output_path", help="Output path to laps.csv.", default="laps.csv", type=str)
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path or "laps.csv"

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    df = pd.read_csv(input_path)
    summary = summarize_laps(df)
    summary.to_csv(output_path, index=False)
    print(f"Wrote {len(summary)} lap summaries to {output_path}")

if __name__ == "__main__":
    main()
