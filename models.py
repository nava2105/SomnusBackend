from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional


class RegisterIn(BaseModel):
    email: EmailStr
    user_name: str
    password: str
    confirm_password: str


class LoginIn(BaseModel):
    email: EmailStr
    password: str


class ProfileData(BaseModel):
    user_id: str
    birth_date: str  # yyyy-mm-dd
    age: int = Field(ge=0)
    profile_picture: str


class SleepTimesData(BaseModel):
    user_id: str
    sleep_time: str  # HH:MM in 24-hour format
    wakeup_time: str  # HH:MM in 24-hour format


class NightPoint(BaseModel):
    time: str
    value: int = Field(ge=0, le=1)


class NightEvent(BaseModel):
    time: str
    type: str  # pickup | wakeup


class NightPatternIn(BaseModel):
    points: List[NightPoint]
    events: List[NightEvent]


class RecommendationIn(BaseModel):
    recommendation_id: str


class Song(BaseModel):
    title: str
    composer: str
    mp3_url: str


class SongRecommendationRequest(BaseModel):
    song_title: str = Field(..., description="Title of the song to get recommendations for")
    k: int = Field(default=50, ge=1, le=100, description="Number of similar songs to search for")
    max_results: int = Field(default=20, ge=1, le=50, description="Maximum number of results to return")


class SongRecommendation(BaseModel):
    title: str
    composer: str
    form: str
    mp3_url: str
    similarity_score: float


class SleepRecommendation(BaseModel):
    id: str
    title: str
    brief_description: str
    detailed_description: str
    embedding: Optional[List[float]] = None


class SimilarSleepRecommendation(BaseModel):
    id: str
    title: str
    brief_description: str
    detailed_description: str
    similarity_score: float


class SleepRecommendationRequest(BaseModel):
    recommendation_titles: List[str] = Field(..., description="List of recommendation titles to find similar ones for")
    k: int = Field(default=50, ge=1, le=100, description="Number of similar recommendations to search for")
    max_results: int = Field(default=20, ge=1, le=50, description="Maximum number of results to return")


class SleepSessionData(BaseModel):
    user_id: str
    startTime: int
    endTime: int
    duration: int
    audioData: List[dict]
    motionData: List[dict]
    lightData: List[dict]
    summary: dict


class DayData(BaseModel):
    day: str
    asleep: float
    awake: float
    pickups: float


class MonthlyDayData(BaseModel):
    date: str
    score: int
    day: int


class SleepDistribution(BaseModel):
    awake: float
    pickups: float
    asleep: float


class SleepScore(BaseModel):
    score: int
    label: str