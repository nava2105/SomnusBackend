import os
from datetime import datetime, timedelta
from jose import jwt
from dotenv import load_dotenv
import uvicorn
from fastapi import FastAPI, HTTPException, status, Query
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import json
import base64
from pathlib import Path
import redis
import random
import numpy as np
from db import users
from auth import hash_password, verify_password, create_token

load_dotenv()

app = FastAPI()

# Enable CORS for Expo frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

r = redis.Redis(host="localhost", port=6379, decode_responses=False)

def get_random_songs(n=5):
    """
    Gets n random songs from Redis with their embeddings
    """
    # Get all songs keys:*
    keys = []
    for key in r.scan_iter("song:*"):
        keys.append(key)

    if not keys:
        print("⚠ No songs found in Redis")
        return []

    # Select n random keys without repeating
    selected_keys = random.sample(keys, min(n, len(keys)))

    # Get data for each song
    songs = []
    for key in selected_keys:
        song_data = r.hgetall(key)

        # Decode the embedding of bytes to a list of floats
        embedding_bytes = song_data[b"embedding"]
        embedding_array = np.frombuffer(embedding_bytes, dtype=np.float32)

        songs.append({
            "title": song_data[b"title"].decode(),
            "composer": song_data[b"composer"].decode(),
            "form": song_data[b"form"].decode(),
            "period": song_data[b"period"].decode(),
            "mp3_url": song_data[b"mp3_url"].decode(),
            "embedding": embedding_array.tolist()
        })

    return songs


def recommend(song_embedding, k=50, max_results=20):
    query_vector = np.array(song_embedding, dtype=np.float32).tobytes()

    q = (
            "*=>[KNN %d @embedding $vec AS score]"
            % k
    )

    result = r.execute_command(
        "FT.SEARCH", "idx:songs", q,
        "PARAMS", "2", "vec", query_vector,
        "SORTBY", "score",
        "RETURN", "5", "title", "composer", "form", "mp3_url", "score",
        "LIMIT", "0", str(max_results),
        "DIALECT", "2"
    )

    return result


def get_total_sleep_recommendations():
    """Get the total number of sleep recommendations in the index"""
    try:
        info = r.execute_command("FT.INFO", "idx:sleep_recommendations")
        for i in range(0, len(info), 2):
            if info[i] == b'num_docs':
                return int(info[i + 1])
        return 0
    except Exception as e:
        print(f"Error getting total recommendations: {e}")
        return 0


def get_random_sleep_recommendations(n=5):
    """Get n random sleep recommendations from Redis"""
    keys = []
    for key in r.scan_iter("sleep:*"):
        keys.append(key)

    if not keys:
        print("⚠️ No sleep recommendations found in Redis")
        return []

    selected_keys = random.sample(keys, min(n, len(keys)))
    recommendations = []

    for key in selected_keys:
        data = r.hgetall(key)
        # Extract ID from key (e.g., "sleep:123" -> "123")
        rec_id = key.decode() if isinstance(key, bytes) else key
        rec_id = rec_id.split(":")[-1]  # Get the number after "sleep:"

        recommendations.append({
            "id": rec_id,
            "title": data[b"title"].decode(),
            "brief_description": data[b"brief_description"].decode(),
            "detailed_description": data[b"detailed_description"].decode()
        })

    return recommendations


def search_similar_recommendations(embedding_query, k=None, max_results=None):
    """Search for similar sleep recommendations using vector similarity"""
    total_docs = get_total_sleep_recommendations()
    if k is None:
        k = total_docs if total_docs > 0 else 1000
    if max_results is None:
        max_results = total_docs if total_docs > 0 else 1000

    query_vector = np.array(embedding_query, dtype=np.float32).tobytes()
    q = "*=>[KNN %d @embedding $vec AS score]" % k

    result = r.execute_command(
        "FT.SEARCH", "idx:sleep_recommendations", q,
        "PARAMS", "2", "vec", query_vector,
        "SORTBY", "score",
        "RETURN", "4", "title", "brief_description", "detailed_description", "score",
        "LIMIT", "0", str(max_results),
        "DIALECT", "2"
    )

    return result


def parse_recommendation_search_results(result, exclude_titles=None):
    """Parse Redis search results into recommendation objects"""
    recommendations = []

    if exclude_titles is None:
        exclude_titles = []
    elif isinstance(exclude_titles, str):
        exclude_titles = [exclude_titles]

    if result and len(result) > 1:
        i = 1
        while i < len(result):
            key = result[i]
            i += 1

            # Extract ID from key
            rec_id = key.decode() if isinstance(key, bytes) else key
            rec_id = rec_id.split(":")[-1]

            if i < len(result) and isinstance(result[i], list):
                fields = result[i]
                i += 1

                field_dict = {}
                for j in range(0, len(fields), 2):
                    if j + 1 < len(fields):
                        field_name = fields[j]
                        field_value = fields[j + 1]

                        if isinstance(field_name, bytes):
                            field_name = field_name.decode()
                        if isinstance(field_value, bytes):
                            field_value = field_value.decode()

                        field_dict[field_name] = field_value

                title = field_dict.get("title", "")
                if title in exclude_titles:
                    continue

                recommendations.append({
                    "id": rec_id,
                    "title": title,
                    "brief_description": field_dict.get("brief_description", ""),
                    "detailed_description": field_dict.get("detailed_description", ""),
                    "score": float(field_dict.get("score", 0))
                })
            else:
                continue

    return recommendations


# AUTH MODELS
class RegisterIn(BaseModel):
    name: str = Field(..., min_length=2, max_length=60)
    password: str = Field(..., min_length=6, max_length=128)
    confirm_password: str = Field(..., min_length=6, max_length=128)


class LoginIn(BaseModel):
    name: str = Field(..., min_length=2, max_length=60)
    password: str = Field(..., min_length=6, max_length=128)


# SONG MODELS
class Song(BaseModel):
    title: str
    composer: str  # Author of the song
    mp3_url: str   # URL to the audio file


# USER DATA MODELS
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
    username: str
    startTime: int
    endTime: int
    duration: int
    audioData: List[dict]   # snore / noise events
    motionData: List[dict]  # movement events
    lightData: List[dict]
    summary: dict


class SongRecommendationRequest(BaseModel):
    song_title: str = Field(..., description="Title of the song to get recommendations for")
    k: int = Field(default=50, ge=1, le=100, description="Number of similar songs to search for")
    max_results: int = Field(default=20, ge=1, le=50, description="Maximum number of results to return")


class SongRecommendation(BaseModel):
    title: str
    composer: str
    form: str
    period: str
    mp3_url: str
    similarity_score: float


class SleepRecommendation(BaseModel):
    id: str  # ADD THIS FIELD
    title: str
    brief_description: str
    detailed_description: str
    embedding: Optional[List[float]] = None


class SimilarSleepRecommendation(BaseModel):
    id: str  # ADD THIS FIELD
    title: str
    brief_description: str
    detailed_description: str  # ADD THIS FIELD
    similarity_score: float


class SleepRecommendationRequest(BaseModel):
    recommendation_titles: List[str] = Field(..., description="List of recommendation titles to find similar ones for")
    k: int = Field(default=50, ge=1, le=100, description="Number of similar recommendations to search for")
    max_results: int = Field(default=20, ge=1, le=50, description="Maximum number of results to return")


# ENDPOINTS

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


@app.post("/auth/register", status_code=201)
async def register(payload: RegisterIn):
    name = payload.name.strip().lower()

    if payload.password != payload.confirm_password:
        raise HTTPException(status_code=400, detail="Las contraseñas no coinciden")

    existing = await users.find_one({"name": name})
    if existing:
        raise HTTPException(status_code=409, detail="El usuario ya existe")

    user_doc = {
        "name": name,
        "password_hash": hash_password(payload.password),
        "created_at": datetime.utcnow()
    }

    await users.insert_one(user_doc)

    return {"status": "success", "message": "Usuario registrado correctamente"}


@app.post("/auth/login")
async def login(payload: LoginIn):
    name = payload.name.strip().lower()

    user = await users.find_one({"name": name})
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Credenciales incorrectas")

    if not verify_password(payload.password, user["password_hash"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Credenciales incorrectas")

    token = create_token(subject=name)

    return {"access_token": token, "token_type": "bearer"}


@app.post("/api/songs/recommend", response_model=List[SongRecommendation])
async def get_song_recommendations(request: SongRecommendationRequest):
    """
    Get song recommendations based on vector similarity using a selected song as reference.
    Uses Redis vector search to find songs with similar musical characteristics.
    """
    try:
        # First, find the reference song in Redis
        reference_song = None
        for key in r.scan_iter("song:*"):
            song_data = r.hgetall(key)
            if song_data[b"title"].decode() == request.song_title:
                # Decode the embedding
                embedding_bytes = song_data[b"embedding"]
                embedding_array = np.frombuffer(embedding_bytes, dtype=np.float32)
                reference_song = {
                    "title": song_data[b"title"].decode(),
                    "embedding": embedding_array.tolist()
                }
                break

        if not reference_song:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Song '{request.song_title}' not found in database"
            )

        # Use the recommend function to find similar songs
        result = recommend(
            song_embedding=reference_song["embedding"],
            k=request.k,
            max_results=request.max_results
        )

        # Parse the Redis search results
        # Format: [total_results, key1, [field1, value1, field2, value2, ...], key2, [...], ...]
        recommendations = []

        if result and len(result) > 1:
            total_results = result[0]
            i = 1

            while i < len(result):
                # Get the key
                key = result[i]
                i += 1

                # Get the fields array
                if i < len(result) and isinstance(result[i], list):
                    fields = result[i]
                    i += 1

                    # Parse fields into a dictionary
                    field_dict = {}
                    for j in range(0, len(fields), 2):
                        if j + 1 < len(fields):
                            field_name = fields[j]
                            field_value = fields[j + 1]

                            # Decode bytes to string
                            if isinstance(field_name, bytes):
                                field_name = field_name.decode()
                            if isinstance(field_value, bytes):
                                field_value = field_value.decode()

                            field_dict[field_name] = field_value

                    # Extract the score (it's in the field list with the alias "score")
                    score = float(field_dict.get("score", 0))

                    # Create recommendation object
                    recommendation = {
                        "title": field_dict.get("title", ""),
                        "composer": field_dict.get("composer", ""),
                        "form": field_dict.get("form", ""),
                        "period": field_dict.get("period", ""),
                        "mp3_url": field_dict.get("mp3_url", ""),
                        "similarity_score": score
                    }

                    # Filter out the reference song itself
                    if recommendation["title"] != request.song_title:
                        recommendations.append(recommendation)
                else:
                    # Skip if format is unexpected
                    continue

        return recommendations

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating recommendations: {str(e)}"
        )


# Optional: Add this helper endpoint to get available songs for recommendation

@app.get("/api/songs/available", response_model=List[Song])
async def get_available_songs(
        limit: int = Query(default=50, ge=1, le=200, description="Maximum number of songs to return")
):
    """
    Get list of available songs in the database for recommendation purposes
    """
    try:
        songs = []
        count = 0

        for key in r.scan_iter("song:*"):
            if count >= limit:
                break

            song_data = r.hgetall(key)
            songs.append({
                "title": song_data[b"title"].decode(),
                "composer": song_data[b"composer"].decode(),
                "mp3_url": song_data[b"mp3_url"].decode(),
            })
            count += 1

        return songs

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving songs: {str(e)}"
        )


@app.get("/api/songs/random", response_model=List[Song])
async def get_random_songs_endpoint():
    """
    Get all songs from the Redis database in random order
    Returns: List of all songs with title, composer (author), and mp3_url
    """
    try:
        keys = []
        for key in r.scan_iter("song:*"):
            keys.append(key)

        if not keys:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No songs found in database"
            )

        # Shuffle all keys and select 5
        selected_keys = random.sample(keys, min(5, len(keys)))

        songs = []
        for key in selected_keys:
            song_data = r.hgetall(key)
            songs.append({
                "title": song_data[b"title"].decode(),
                "composer": song_data[b"composer"].decode(),
                "mp3_url": song_data[b"mp3_url"].decode(),
            })

        return songs

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving songs: {str(e)}"
        )


@app.get("/api/sleep/recommendations", response_model=List[Recommendation])
async def get_recommendations():
    """
    Get all sleep recommendations
    """
    return [
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


@app.get("/api/sleep/weekly-data", response_model=List[DayData])
async def get_weekly_data():
    """
    Get weekly sleep data for bar chart
    Returns: Array of daily sleep metrics
    """
    return [
        {"day": "Sun", "asleep": 40, "awake": 58, "pickups": 2},
        {"day": "Mon", "asleep": 25, "awake": 67, "pickups": 8},
        {"day": "Tue", "asleep": 35, "awake": 60, "pickups": 5},
        {"day": "Wed", "asleep": 33, "awake": 53, "pickups": 14},
        {"day": "Thu", "asleep": 21, "awake": 78, "pickups": 1},
        {"day": "Fri", "asleep": 50, "awake": 46, "pickups": 4},
        {"day": "Sat", "asleep": 35, "awake": 62, "pickups": 3},
    ]


@app.get("/api/sleep/monthly-data", response_model=List[MonthlyDayData])
async def get_monthly_data():
    """
    Get monthly sleep data for calendar view
    Returns: Array of daily sleep scores
    """
    return [
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


@app.get("/api/sleep/sleep-distribution", response_model=SleepDistribution)
async def get_sleep_distribution():
    """
    Get sleep phase distribution for pie/arc chart
    Returns: Percentages for awake, pickups, and asleep time
    """
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
    return {
        "points": [
            {"time": "22:00", "value": 0}, {"time": "22:05", "value": 0.1}, {"time": "22:10", "value": 0.2},
            {"time": "22:15", "value": 0.4}, {"time": "22:20", "value": 0.6}, {"time": "22:25", "value": 0.8},
            {"time": "22:30", "value": 1.0}, {"time": "22:35", "value": 1.0}, {"time": "22:40", "value": 0.9},
            {"time": "22:45", "value": 0.95}, {"time": "22:50", "value": 1.0}, {"time": "22:55", "value": 1.0},
            {"time": "23:00", "value": 0.9}, {"time": "23:05", "value": 1.0}, {"time": "23:10", "value": 0.8},
            {"time": "23:15", "value": 0.6}, {"time": "23:20", "value": 0.7}, {"time": "23:25", "value": 0.9},
            {"time": "23:30", "value": 1.0}, {"time": "23:35", "value": 0.95}, {"time": "23:40", "value": 0.7},
            {"time": "23:45", "value": 0.4}, {"time": "23:50", "value": 0.6}, {"time": "23:55", "value": 0.8},
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


@app.post("/api/sleep/session")
async def receive_sleep_session(session: SleepSessionData):
    print("\n" + "=" * 50)
    print("SLEEP SESSION RECEIVED")
    print(f"User: {session.username}")
    print(f"Duration: {session.duration / 1000 / 60:.2f} min")
    print(f"Snores: {len(session.audioData)}")
    print(f"Movements: {len(session.motionData)}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = Path(f"data/{session.username}/{timestamp}")
    session_dir.mkdir(parents=True, exist_ok=True)

    with open(session_dir / "audio_events.json", "w") as f:
        json.dump(session.audioData, f, indent=2)

    with open(session_dir / "motion_events.json", "w") as f:
        json.dump(session.motionData, f, indent=2)

    with open(session_dir / "summary.json", "w") as f:
        json.dump(session.summary, f, indent=2)

    print("Session saved ✔")
    print("=" * 50)

    return {"status": "success"}


@app.get("/api/sleep/random", response_model=List[SleepRecommendation])
async def get_all_sleep_recommendations_random():
    """
    Get ALL sleep recommendations from Redis in random order
    Returns: List of all sleep recommendations with id, title and descriptions in random order
    """
    try:
        keys = []
        for key in r.scan_iter("sleep:*"):
            keys.append(key)

        if not keys:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No sleep recommendations found in database"
            )

        # Shuffle all keys
        random.shuffle(keys)

        recommendations = []
        for key in keys:
            data = r.hgetall(key)
            # Extract ID from key (e.g., "sleep:123" -> "123")
            rec_id = key.decode() if isinstance(key, bytes) else key
            rec_id = rec_id.split(":")[-1]

            recommendations.append({
                "id": rec_id,
                "title": data[b"title"].decode(),
                "brief_description": data[b"brief_description"].decode(),
                "detailed_description": data[b"detailed_description"].decode()
            })

        return recommendations

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving recommendations: {str(e)}"
        )


@app.post("/api/sleep/recommend", response_model=List[SimilarSleepRecommendation])
async def get_similar_sleep_recommendations(request: SleepRecommendationRequest):
    """
    Get sleep recommendations similar to multiple selected recommendations using vector similarity
    Uses the average of all selected recommendation embeddings to find similar ones
    """
    try:
        # Find all reference recommendations
        reference_embeddings = []
        reference_titles_found = []

        for title in request.recommendation_titles:
            for key in r.scan_iter("sleep:*"):
                data = r.hgetall(key)
                if data[b"title"].decode() == title:
                    embedding_bytes = data[b"embedding"]
                    embedding_array = np.frombuffer(embedding_bytes, dtype=np.float32)
                    reference_embeddings.append(embedding_array)
                    reference_titles_found.append(title)
                    break

        if not reference_embeddings:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"None of the provided recommendations were found"
            )

        # Average all embeddings to create composite query vector
        composite_embedding = np.mean(reference_embeddings, axis=0).tolist()

        total_recommendations = get_total_sleep_recommendations()

        # Search for similar recommendations
        result = search_similar_recommendations(
            embedding_query=composite_embedding,
            k=total_recommendations * 2 if total_recommendations > 0 else 1000,
            max_results=total_recommendations if total_recommendations > 0 else 1000
        )

        # Parse results with all reference titles excluded
        recommendations = parse_recommendation_search_results(
            result,
            exclude_titles=request.recommendation_titles
        )

        # Format for response - NOW INCLUDING ID
        formatted_recommendations = [
            {
                "id": rec["id"],
                "title": rec["title"],
                "brief_description": rec["brief_description"],
                "detailed_description": rec["detailed_description"],
                "similarity_score": rec["score"]
            }
            for rec in recommendations
        ]

        return formatted_recommendations

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating similar recommendations: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)