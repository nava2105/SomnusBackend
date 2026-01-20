import joblib
import pandas as pd
from typing import List

# Load ML model
try:
    model = joblib.load("sleep_quality_model.pkl")
    print("✅ ML model loaded successfully")
except Exception as e:
    print(f"⚠️ Failed to load ML model: {e}")
    model = None


def count_pickups(motion_events: List[dict], movement_threshold: float = 1.8, min_gap_ms: int = 30_000):
    if not motion_events:
        return 0

    motion_events = sorted(motion_events, key=lambda e: e["time"])

    pickups = 0
    last_pickup_time = None

    for event in motion_events:
        movement = event.get("movement", 0)
        time = event["time"]

        if movement >= movement_threshold:
            if last_pickup_time is None or time - last_pickup_time > min_gap_ms:
                pickups += 1
                last_pickup_time = time

    return pickups



def compute_distribution(duration_ms: int, pickup_count: int):
    """
    Compute sleep distribution percentages based on 24 hours
    Returns: (asleep_pct, awake_pct, pickups_pct)
    """
    DAY_MS = 24 * 60 * 60 * 1000

    sleep_ms = min(duration_ms, DAY_MS)
    pickup_time_ms = pickup_count * 2 * 60 * 1000  # assume 2 min per pickup

    asleep_pct = (sleep_ms / DAY_MS) * 100
    pickups_pct = (pickup_time_ms / DAY_MS) * 100
    awake_pct = max(0, 100 - asleep_pct - pickups_pct)

    return (
        round(asleep_pct, 2),
        round(awake_pct, 2),
        round(pickups_pct, 2),
    )


def score_to_label(score: int) -> str:
    """
    Convert numeric sleep score to descriptive label
    """
    if score >= 80:
        return "Excellent Sleep"
    elif score >= 60:
        return "Good Sleep"
    elif score >= 40:
        return "Fragmented Sleep"
    elif score >= 20:
        return "Poor Sleep"
    return "Very Poor Sleep"


def calculate_sleep_score(audio_data: List[dict], motion_data: List[dict], duration: int) -> int:
    """
    Calculate sleep quality score using ML model
    Returns score between 0-100
    """
    if model is None:
        return 50  # Default score if model not available

    try:
        features = {
            "audio_event_rate": len(audio_data) / max(1, duration / 1000),
            "audio_intensity_mean": sum(e.get("intensity", 0) for e in audio_data) / max(1, len(audio_data)),
            "audio_intensity_max": max([e.get("intensity", 0) for e in audio_data], default=0),
            "audio_freq_mean": sum(e.get("frequency", 0) for e in audio_data) / max(1, len(audio_data)),
            "motion_event_rate": len(motion_data) / max(1, duration / 1000),
            "motion_intensity_mean": (sum(m.get("movement", 0) for m in motion_data) / max(1, len(motion_data))),
            "duration_sec": duration / 1000,
        }

        score = int(round(model.predict(pd.DataFrame([features]))[0]))
        score = max(0, min(100, score))  # Clamp between 0-100

        print(f"🤖 ML Score: {score}")
        return score

    except Exception as e:
        print(f"⚠️ ML scoring failed, using default: {e}")
        return 50