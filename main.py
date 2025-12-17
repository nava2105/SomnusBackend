from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import List
from datetime import datetime
import os
import json
import base64
from pathlib import Path

app = FastAPI()

# Enable CORS for Expo frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for data validation
class UserTimeSetting(BaseModel):
    hour: int
    minute: int
    isPM: bool


class UserProfile(BaseModel):
    username: str
    profile_picture: str
    age: int
    birth_date: str


class UserSettings(BaseModel):
    username: str
    bedtime: UserTimeSetting
    wakeup_time: UserTimeSetting


class Recommendation(BaseModel):
    id: str
    title: str
    brief_explanation: str
    detailed_explanation: str


class PinAction(BaseModel):
    username: str
    recommendation_id: str
    recommendation_title: str
    pinned: bool


class DayData(BaseModel):
    day: str
    asleep: int
    awake: int
    pickups: int


class MonthlyDayData(BaseModel):
    date: str
    score: int
    day: int


class NightGraphPoint(BaseModel):
    time: str
    value: float


class NightGraphEvent(BaseModel):
    time: str
    type: str


class SleepDistribution(BaseModel):
    awake: int
    pickups: int
    asleep: int


class SleepScore(BaseModel):
    score: int
    label: str


class SleepSessionData(BaseModel):
    username: str  # Added username field
    startTime: str
    endTime: str
    duration: int
    audioData: List[dict]
    motionData: List[dict]
    lightData: List[dict]
    summary: dict


# In-memory storage (replace with database in production)
recommendations_db = [
    {
        "id": "1",
        "title": "Maintain Consistent Sleep Schedule",
        "brief_explanation": "Going to bed and waking up at the same time helps regulate your circadian rhythm.",
        "detailed_explanation": "Your body has an internal clock called the circadian rhythm. When you maintain consistent sleep times, you reinforce this rhythm, making it easier to fall asleep and wake up naturally. This can improve sleep quality by up to 23%."
    },
    {
        "id": "2",
        "title": "Reduce Screen Time Before Bed",
        "brief_explanation": "Avoid phones and tablets 1 hour before bedtime to improve melatonin production.",
        "detailed_explanation": "Blue light from screens suppresses melatonin production, the hormone that regulates sleep. Studies show that reducing screen time 1-2 hours before bed can help you fall asleep 30% faster and increase REM sleep duration."
    },
    {
        "id": "3",
        "title": "Optimize Room Temperature",
        "brief_explanation": "Keep your bedroom between 65-68°F (18-20°C) for optimal sleep.",
        "detailed_explanation": "Core body temperature naturally drops during sleep. A cooler room helps facilitate this process. Research indicates that temperatures between 65-68°F promote deeper sleep and reduce nighttime awakenings."
    },
]

# Mock sleep data
weekly_sleep_data = [
    {"day": "Sun", "asleep": 40, "awake": 58, "pickups": 2},
    {"day": "Mon", "asleep": 25, "awake": 67, "pickups": 8},
    {"day": "Tue", "asleep": 35, "awake": 60, "pickups": 5},
    {"day": "Wed", "asleep": 33, "awake": 53, "pickups": 14},
    {"day": "Thu", "asleep": 21, "awake": 78, "pickups": 1},
    {"day": "Fri", "asleep": 50, "awake": 46, "pickups": 4},
    {"day": "Sat", "asleep": 35, "awake": 62, "pickups": 3},
]

monthly_sleep_data = [
    # September 2025 data
    {"date": "2025-11-01", "score": 85, "day": 1},
    {"date": "2025-11-02", "score": 92, "day": 2},
    {"date": "2025-11-03", "score": 78, "day": 3},
    {"date": "2025-11-04", "score": 65, "day": 4},
    {"date": "2025-11-05", "score": 88, "day": 5},
    {"date": "2025-11-06", "score": 95, "day": 6},
    {"date": "2025-11-07", "score": 82, "day": 7},
    {"date": "2025-11-08", "score": 91, "day": 8},
    {"date": "2025-11-09", "score": 73, "day": 9},
    {"date": "2025-11-10", "score": 55, "day": 10},
    {"date": "2025-11-11", "score": 89, "day": 11},
    {"date": "2025-11-12", "score": 86, "day": 12},
    {"date": "2025-11-13", "score": 93, "day": 13},
    {"date": "2025-11-14", "score": 79, "day": 14},
    {"date": "2025-11-15", "score": 68, "day": 15},
    {"date": "2025-11-16", "score": 84, "day": 16},
    {"date": "2025-11-17", "score": 77, "day": 17},
    {"date": "2025-11-18", "score": 90, "day": 18},
    {"date": "2025-11-19", "score": 88, "day": 19},
    {"date": "2025-11-20", "score": 52, "day": 20},
    {"date": "2025-11-21", "score": 83, "day": 21},
    {"date": "2025-11-22", "score": 76, "day": 22},
    {"date": "2025-11-23", "score": 94, "day": 23},
    {"date": "2025-11-24", "score": 81, "day": 24},
    {"date": "2025-11-25", "score": 71, "day": 25},
    {"date": "2025-11-26", "score": 87, "day": 26},
    {"date": "2025-11-27", "score": 69, "day": 27},
    {"date": "2025-11-28", "score": 58, "day": 28},
    {"date": "2025-11-29", "score": 85, "day": 29},
    {"date": "2025-11-30", "score": 92, "day": 30},
    {"date": "2025-12-01", "score": 66, "day": 1},
    {"date": "2025-12-02", "score": 63, "day": 2},
    {"date": "2025-12-03", "score": 70, "day": 3},
    {"date": "2025-12-04", "score": 78, "day": 4},
    {"date": "2025-12-05", "score": 88, "day": 5},
    {"date": "2025-12-06", "score": 81, "day": 6},
    {"date": "2025-12-07", "score": 59, "day": 7},
    {"date": "2025-12-08", "score": 79, "day": 8},
    {"date": "2025-12-09", "score": 89, "day": 9},
    {"date": "2025-12-10", "score": 85, "day": 10},
    {"date": "2025-12-11", "score": 95, "day": 11},
    {"date": "2025-12-12", "score": 79, "day": 12},
    {"date": "2025-12-13", "score": 74, "day": 13},
    {"date": "2025-12-14", "score": 69, "day": 14},
    {"date": "2025-12-15", "score": 81, "day": 15},
]

night_pattern_data = {
    "points": [
        # Initial period - getting to sleep
        {"time": "22:00", "value": 0}, {"time": "22:05", "value": 0.1}, {"time": "22:10", "value": 0.2},
        {"time": "22:15", "value": 0.4}, {"time": "22:20", "value": 0.6}, {"time": "22:25", "value": 0.8},
        # Deep sleep phase
        {"time": "22:30", "value": 1.0}, {"time": "22:35", "value": 1.0}, {"time": "22:40", "value": 0.9},
        {"time": "22:45", "value": 0.95}, {"time": "22:50", "value": 1.0}, {"time": "22:55", "value": 1.0},
        {"time": "23:00", "value": 0.9}, {"time": "23:05", "value": 1.0}, {"time": "23:10", "value": 0.8},
        # REM/light sleep cycle
        {"time": "23:15", "value": 0.6}, {"time": "23:20", "value": 0.7}, {"time": "23:25", "value": 0.9},
        {"time": "23:30", "value": 1.0}, {"time": "23:35", "value": 0.95}, {"time": "23:40", "value": 0.7},
        {"time": "23:45", "value": 0.4}, {"time": "23:50", "value": 0.6}, {"time": "23:55", "value": 0.8},
        # Continued through the night...
        {"time": "00:00", "value": 1.0}, {"time": "00:30", "value": 0.9}, {"time": "01:00", "value": 1.0},
        {"time": "01:30", "value": 0.7}, {"time": "02:00", "value": 0.8}, {"time": "02:30", "value": 1.0},
        {"time": "03:00", "value": 0.9}, {"time": "03:30", "value": 0.6}, {"time": "04:00", "value": 0.7},
        {"time": "04:30", "value": 0.8}, {"time": "05:00", "value": 0.9}, {"time": "05:30", "value": 0.6},
        {"time": "06:00", "value": 0.4}, {"time": "06:30", "value": 0.2}, {"time": "07:00", "value": 0},
    ],
    "events": [
        {"time": "03:55", "type": "pickup"},
        {"time": "10:11", "type": "wakeup"},
    ]
}


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


@app.get("/health")
async def health_check():
    """Health check endpoint to verify API is running"""
    return {"status": "healthy", "message": "API is running"}


@app.get("/api/recommendations", response_model=List[Recommendation])
async def get_recommendations():
    """
    Get all sleep recommendations
    """
    return recommendations_db


@app.get("/api/sleep/weekly-data", response_model=List[DayData])
async def get_weekly_data():
    """
    Get weekly sleep data for bar chart
    Returns: Array of daily sleep metrics
    """
    return weekly_sleep_data


@app.get("/api/sleep/monthly-data", response_model=List[MonthlyDayData])
async def get_monthly_data():
    """
    Get monthly sleep data for calendar view
    Returns: Array of daily sleep scores
    """
    return monthly_sleep_data


@app.get("/api/sleep/sleep-distribution", response_model=SleepDistribution)
async def get_sleep_distribution():
    """
    Get sleep phase distribution for pie/arc chart
    Returns: Percentages for awake, pickups, and asleep time
    """
    # In production, calculate this from actual data
    return {
        "awake": 60,
        "pickups": 5,
        "asleep": 35
    }


@app.get("/api/sleep/score", response_model=SleepScore)
async def get_sleep_score():
    """
    Get current sleep score
    Returns: Score value and label
    """
    return {
        "score": 85,
        "label": "Night Score"
    }


@app.get("/api/sleep/night-pattern")
async def get_night_pattern():
    """
    Get detailed night sleep pattern data
    Returns: Graph points and events
    """
    return night_pattern_data


@app.post("/api/user/profile")
async def receive_user_profile(profile: UserProfile):
    """
    Receive user profile data from frontend
    """
    print("\n" + "=" * 50)
    print("RECEIVED USER PROFILE DATA:")
    print(f"Username: {profile.username}")
    print(f"Profile Picture: {profile.profile_picture}")
    print(f"Age: {profile.age}")
    print(f"Birth Date: {profile.birth_date}")
    print("=" * 50 + "\n")

    return {"status": "success", "message": "Profile data received"}


@app.post("/api/user/settings")
async def receive_user_settings(settings: UserSettings):
    """
    Receive user sleep settings from frontend
    """
    print("\n" + "=" * 50)
    print("RECEIVED USER SETTINGS DATA:")
    print(f"Username: {settings.username}")
    bedtime = f"{settings.bedtime.hour}:{settings.bedtime.minute:02d} {'PM' if settings.bedtime.isPM else 'AM'}"
    wakeup = f"{settings.wakeup_time.hour}:{settings.wakeup_time.minute:02d} {'PM' if settings.wakeup_time.isPM else 'AM'}"
    print(f"Bedtime: {bedtime}")
    print(f"Wake-up Time: {wakeup}")
    print("=" * 50 + "\n")

    return {"status": "success", "message": "Settings data received"}


@app.post("/api/user/pin-recommendation")
async def pin_recommendation(pin_action: PinAction):
    """
    Log when a user pins/unpins a recommendation
    """
    print("\n" + "=" * 50)
    print(f"USER PIN ACTION:")
    print(f"Username: {pin_action.username}")
    print(f"Recommendation: {pin_action.recommendation_title} (ID: {pin_action.recommendation_id})")
    print(f"Action: {'PINNED' if pin_action.pinned else 'UNPINNED'}")
    print("=" * 50 + "\n")

    return {
        "status": "success",
        "message": "Pin action logged",
        "action": pin_action
    }


@app.post("/api/sleep/session")
async def receive_sleep_session(session: SleepSessionData):
    """
    Receive complete sleep session data from frontend and save to disk
    """
    print("\n" + "=" * 50)
    print("RECEIVED SLEEP SESSION DATA:")
    print(f"Username: {session.username}")
    print(f"Start Time: {session.startTime}")
    print(f"End Time: {session.endTime}")
    print(f"Duration: {session.duration / 1000 / 60:.2f} minutes")
    print(f"Motion Events: {len(session.motionData)}")
    print(f"Light Readings: {len(session.lightData)}")

    # Create directory structure: data/{username}/{timestamp}/
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = Path(f"data/{session.username}/{timestamp}")
    session_dir.mkdir(parents=True, exist_ok=True)

    # Save audio file if present
    audio_saved = False
    if session.audioData and len(session.audioData) > 0:
        audio_entry = session.audioData[0]
        audio_base64 = audio_entry.get('audioFileBase64', '')
        if audio_base64:
            try:
                audio_path = session_dir / "audio.m4a"
                with open(audio_path, "wb") as f:
                    f.write(base64.b64decode(audio_base64))
                audio_saved = True
                print(f"Audio file saved: {audio_path}")
            except Exception as e:
                print(f"Failed to save audio file: {e}")

    # Save motion data
    try:
        motion_path = session_dir / "motion.json"
        with open(motion_path, "w") as f:
            json.dump(session.motionData, f, indent=2)
        print(f"Motion data saved: {motion_path}")
    except Exception as e:
        print(f"Failed to save motion data: {e}")

    # Save light data
    try:
        light_path = session_dir / "light.json"
        with open(light_path, "w") as f:
            json.dump(session.lightData, f, indent=2)
        print(f"Light data saved: {light_path}")
    except Exception as e:
        print(f"Failed to save light data: {e}")

    # Save session metadata
    try:
        session_path = session_dir / "session.json"
        session_metadata = {
            "startTime": session.startTime,
            "endTime": session.endTime,
            "duration": session.duration,
            "summary": session.summary,
            "username": session.username
        }
        with open(session_path, "w") as f:
            json.dump(session_metadata, f, indent=2)
        print(f"Session metadata saved: {session_path}")
    except Exception as e:
        print(f"Failed to save session metadata: {e}")

    print(f"Audio File: {'Yes' if audio_saved else 'No'}")
    print("=" * 50 + "\n")

    return {
        "status": "success",
        "message": "Sleep session received and saved to disk",
        "session_id": f"session_{datetime.now().isoformat()}",
        "files_saved": {
            "audio": audio_saved,
            "motion": True,
            "light": True,
            "session": True
        },
        "directory": str(session_dir)
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)