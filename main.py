from datetime import datetime, timezone
from dotenv import load_dotenv
import uvicorn
from fastapi import FastAPI, HTTPException, status, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import random
import numpy as np

from db import users
from auth import hash_password, verify_password
from uuid import uuid4
from models import (
    RegisterIn, LoginIn, ProfileData, SleepTimesData, NightPatternIn,
    RecommendationIn, Song, SongRecommendationRequest, SongRecommendation,
    SleepRecommendation, SimilarSleepRecommendation, SleepRecommendationRequest,
    SleepSessionData, DayData, MonthlyDayData, SleepDistribution, SleepScore
)
from recommendations import (
    recommend_songs, search_similar_recommendations, parse_recommendation_search_results,
    get_all_songs, get_all_sleep_recommendations_shuffled, find_song_by_title,
    get_user_saved_recommendations, r
)
from sleep_quality import (
    count_pickups, compute_distribution, score_to_label, calculate_sleep_score
)

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== AUTH ENDPOINTS ====================

@app.get("/")
async def root():
    return {"message": "Somnus Sleep Helper API"}


@app.get("/health")
async def health_check():
    """Health check endpoint to verify API is running"""
    return {"status": "healthy", "message": "API is running"}


@app.post("/auth/register", status_code=201)
async def register(payload: RegisterIn):
    if payload.password != payload.confirm_password:
        raise HTTPException(400, "Passwords do not match")

    if await users.find_one({"email": payload.email.lower()}):
        raise HTTPException(409, "Email already exists")

    user = {
        "user_id": str(uuid4()),
        "email": payload.email.lower(),
        "user_name": payload.user_name,
        "password": hash_password(payload.password),
        "profile_picture": "",
        "birth_date": None,
        "age": None,
        "sleep_time": None,
        "wakeup_time": None,
        "scores": [],
        "night_pattern": [],
        "recommendations": [],
        "created_at": datetime.utcnow()
    }

    await users.insert_one(user)
    return {
        "user_id": user["user_id"],
        "user_name": user["user_name"]
    }


@app.post("/auth/login")
async def login(payload: LoginIn):
    user = await users.find_one({"email": payload.email.lower()})
    if not user or not verify_password(payload.password, user["password"]):
        raise HTTPException(401, "Invalid credentials")

    return {
        "user_id": user["user_id"],
        "user_name": user["user_name"]
    }


# ==================== USER PROFILE ENDPOINTS ====================

@app.post("/api/user/profile-data")
async def save_profile_data(payload: ProfileData):
    """Save user profile data (birth date, age, profile picture)"""
    try:
        result = await users.update_one(
            {"user_id": payload.user_id},
            {
                "$set": {
                    "birth_date": payload.birth_date,
                    "age": payload.age,
                    "profile_picture": payload.profile_picture
                }
            }
        )

        if result.matched_count == 0:
            raise HTTPException(404, "User not found")

        return {"status": "success", "message": "Profile data saved"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed to save profile data: {str(e)}")


@app.post("/api/user/sleep-times")
async def save_sleep_times(payload: SleepTimesData):
    """Save user sleep and wakeup times"""
    try:
        result = await users.update_one(
            {"user_id": payload.user_id},
            {
                "$set": {
                    "sleep_time": payload.sleep_time,
                    "wakeup_time": payload.wakeup_time
                }
            }
        )

        if result.matched_count == 0:
            raise HTTPException(404, "User not found")

        return {"status": "success", "message": "Sleep times saved"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed to save sleep times: {str(e)}")


# ==================== SLEEP SESSION ENDPOINTS ====================

@app.post("/api/sleep/session")
async def receive_sleep_session(session: SleepSessionData):
    """Process and save sleep session data with ML scoring"""
    user = await users.find_one({"user_id": session.user_id})
    if not user:
        raise HTTPException(404, "User not found")

    MIN_SLEEP_MS = 15 * 60 * 1000  # 15 minutes

    if session.duration < MIN_SLEEP_MS:
        raise HTTPException(
            status_code=400,
            detail="Sleep session too short to score"
        )

    print("\n" + "=" * 50)
    print("📥 SLEEP SESSION RECEIVED")
    print(f"User ID: {session.user_id}")
    print(f"Duration: {session.duration / 1000 / 60:.2f} min")
    print(f"Audio events: {len(session.audioData)}")
    print(f"Motion events: {len(session.motionData)}")
    print(f"Light readings: {len(session.lightData)}")
    print("=" * 50)

    # Date & day
    end_dt = datetime.fromtimestamp(
        session.endTime / 1000,
        tz=timezone.utc
    )
    date_str = end_dt.strftime("%Y-%m-%d")
    day_str = end_dt.strftime("%a")

    # Sleep duration
    sleep_hours = round(session.duration / 1000 / 60 / 60, 2)

    # Pickups
    pickup_count = count_pickups(session.motionData)

    # Distribution (24h percentages)
    asleep_pct, awake_pct, pickups_pct = compute_distribution(
        session.duration,
        pickup_count
    )

    # ML Model Score
    score = calculate_sleep_score(
        session.audioData,
        session.motionData,
        session.duration
    )

    # Final score entry
    score_entry = {
        "date": date_str,
        "day": day_str,
        "score": score,
        "asleep": asleep_pct,
        "awake": awake_pct,
        "pickups": pickups_pct,
    }

    # Save to database
    await users.update_one(
        {"user_id": session.user_id},
        {"$push": {"scores": score_entry}}
    )

    print(f"✅ Score entry saved: {score_entry}")
    print("=" * 50 + "\n")

    return {
        "status": "saved",
        "score": score_entry
    }


@app.post("/api/register_night")
async def register_night(user_id: str, payload: NightPatternIn):
    """Save night pattern data"""
    await users.update_one(
        {"user_id": user_id},
        {"$push": {"night_pattern": payload.dict()}}
    )

    return {"status": "night_saved"}


# ==================== SLEEP DATA RETRIEVAL ENDPOINTS ====================

@app.get("/api/sleep/score", response_model=SleepScore)
async def get_sleep_score(user_id: str):
    """Return the latest sleep score for the user"""
    user = await users.find_one(
        {"user_id": user_id},
        {"_id": 0, "scores": 1}
    )

    if not user or not user.get("scores"):
        return {"score": 0, "label": "No Data"}

    latest = user["scores"][-1]
    score = int(latest["score"])

    return {
        "score": score,
        "label": score_to_label(score)
    }


@app.get("/api/sleep/weekly-data", response_model=List[DayData])
async def get_weekly_data(user_id: str):
    """Get last 7 days of sleep data for bar chart"""
    user = await users.find_one(
        {"user_id": user_id},
        {"_id": 0, "scores": 1}
    )

    if not user or not user.get("scores"):
        days = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
        return [{"day": d, "asleep": 0, "awake": 0, "pickups": 0} for d in days]

    scores = user["scores"]
    last_7 = scores[-7:] if len(scores) >= 7 else scores

    result = []
    for entry in last_7:
        result.append({
            "day": entry.get("day", ""),
            "asleep": entry.get("asleep", 0),
            "awake": entry.get("awake", 0),
            "pickups": entry.get("pickups", 0)
        })

    return result


@app.get("/api/sleep/monthly-data", response_model=List[MonthlyDayData])
async def get_monthly_data(user_id: str):
    """Get monthly sleep data for calendar view"""
    user = await users.find_one(
        {"user_id": user_id},
        {"_id": 0, "scores": 1}
    )

    if not user or not user.get("scores"):
        return []

    result = []
    for entry in user["scores"]:
        try:
            date_obj = datetime.strptime(entry["date"], "%Y-%m-%d")
            result.append({
                "date": entry["date"],
                "score": entry["score"],
                "day": date_obj.day
            })
        except:
            continue

    return result


@app.get("/api/sleep/sleep-distribution", response_model=SleepDistribution)
async def get_sleep_distribution(user_id: str):
    """Get sleep phase distribution from latest session"""
    user = await users.find_one(
        {"user_id": user_id},
        {"_id": 0, "scores": 1}
    )

    if not user or not user.get("scores"):
        return {"awake": 0, "pickups": 0, "asleep": 0}

    latest = user["scores"][-1]

    return {
        "awake": latest.get("awake", 0),
        "pickups": latest.get("pickups", 0),
        "asleep": latest.get("asleep", 0)
    }


@app.get("/api/sleep/night-pattern")
async def get_night_pattern(user_id: str):
    """Get detailed night sleep pattern data from latest session"""
    user = await users.find_one(
        {"user_id": user_id},
        {"_id": 0, "night_pattern": 1}
    )

    if not user or not user.get("night_pattern"):
        return {"points": [], "events": []}

    latest_pattern = user["night_pattern"][-1]

    return {
        "points": latest_pattern.get("points", []),
        "events": latest_pattern.get("events", [])
    }


# ==================== RECOMMENDATIONS ENDPOINTS ====================

@app.post("/api/user/recommendations/add")
async def add_recommendation(user_id: str, payload: RecommendationIn):
    """Add a recommendation to user's saved list"""
    await users.update_one(
        {
            "user_id": user_id,
            "recommendations.recommendation_id": {"$ne": payload.recommendation_id}
        },
        {
            "$push": {
                "recommendations": payload.dict()
            }
        }
    )

    return {"status": "recommendation_added"}


@app.post("/api/user/recommendations/remove")
async def remove_recommendation(user_id: str, payload: RecommendationIn):
    """Remove a recommendation from user's saved list"""
    await users.update_one(
        {"user_id": user_id},
        {
            "$pull": {
                "recommendations": {
                    "recommendation_id": payload.recommendation_id
                }
            }
        }
    )

    return {"status": "recommendation_removed"}


@app.get("/api/user/recommendations", response_model=List[SleepRecommendation])
async def get_user_recommendations(user_id: str):
    """Get user's saved recommendations"""
    user = await users.find_one(
        {"user_id": user_id},
        {"_id": 0, "recommendations": 1}
    )

    if not user:
        raise HTTPException(404, "User not found")

    saved = user.get("recommendations", [])
    if not saved:
        return []

    saved_ids = {rec["recommendation_id"] for rec in saved}

    results = get_user_saved_recommendations(list(saved_ids))

    return results


@app.get("/api/sleep/random", response_model=List[SleepRecommendation])
async def get_all_sleep_recommendations_random():
    """Get all sleep recommendations from Redis in random order"""
    try:
        recommendations = get_all_sleep_recommendations_shuffled()

        if not recommendations:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No sleep recommendations found in database"
            )

        return recommendations

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving recommendations: {str(e)}"
        )


@app.post("/api/sleep/recommend", response_model=List[SimilarSleepRecommendation])
async def get_similar_sleep_recommendations(request: SleepRecommendationRequest):
    """Get sleep recommendations similar to multiple selected recommendations"""
    try:
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

        # Average embeddings to create composite query vector
        composite_embedding = np.mean(reference_embeddings, axis=0).tolist()

        # Search for similar recommendations
        result = search_similar_recommendations(
            embedding_query=composite_embedding,
            k=request.k,
            max_results=request.max_results
        )

        # Parse results with all reference titles excluded
        recommendations = parse_recommendation_search_results(
            result,
            exclude_titles=request.recommendation_titles
        )

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


# ==================== MUSIC ENDPOINTS ====================

@app.get("/api/songs/random", response_model=List[Song])
async def get_random_songs_endpoint():
    """Get 5 random songs from the database"""
    try:
        songs = get_all_songs()

        if not songs:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No songs found in database"
            )

        selected_songs = random.sample(songs, min(5, len(songs)))
        return selected_songs

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving songs: {str(e)}"
        )


@app.post("/api/songs/recommend", response_model=List[SongRecommendation])
async def get_song_recommendations(request: SongRecommendationRequest):
    """Get song recommendations based on vector similarity"""
    try:
        reference_song = find_song_by_title(request.song_title)

        if not reference_song:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Song '{request.song_title}' not found in database"
            )

        result = recommend_songs(
            song_embedding=reference_song["embedding"],
            k=request.k,
            max_results=request.max_results
        )

        # Parse Redis search results
        recommendations = []

        if result and len(result) > 1:
            i = 1

            while i < len(result):
                key = result[i]
                i += 1

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

                    score = float(field_dict.get("score", 0))

                    recommendation = {
                        "title": field_dict.get("title", ""),
                        "composer": field_dict.get("composer", ""),
                        "form": field_dict.get("form", ""),
                        "mp3_url": field_dict.get("mp3_url", ""),
                        "similarity_score": score
                    }

                    if recommendation["title"] != request.song_title:
                        recommendations.append(recommendation)
                else:
                    continue

        return recommendations

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating recommendations: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)