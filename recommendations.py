import redis
import random
import numpy as np
from typing import List, Optional

# Initialize Redis connection
r = redis.Redis(host="localhost", port=6379, decode_responses=False)


def get_random_songs(n=5):
    """
    Gets n random songs from Redis with their embeddings
    """
    keys = []
    for key in r.scan_iter("song:*"):
        keys.append(key)

    if not keys:
        print("⚠ No songs found in Redis")
        return []

    selected_keys = random.sample(keys, min(n, len(keys)))

    songs = []
    for key in selected_keys:
        song_data = r.hgetall(key)

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


def recommend_songs(song_embedding, k=50, max_results=20):
    """
    Find similar songs using vector similarity search
    """
    query_vector = np.array(song_embedding, dtype=np.float32).tobytes()

    q = "*=>[KNN %d @embedding $vec AS score]" % k

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
        rec_id = key.decode() if isinstance(key, bytes) else key
        rec_id = rec_id.split(":")[-1]

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


def get_all_songs(limit=None):
    """Get all songs from Redis"""
    songs = []
    count = 0

    for key in r.scan_iter("song:*"):
        if limit and count >= limit:
            break

        song_data = r.hgetall(key)
        songs.append({
            "title": song_data[b"title"].decode(),
            "composer": song_data[b"composer"].decode(),
            "mp3_url": song_data[b"mp3_url"].decode(),
        })
        count += 1

    return songs


def get_all_sleep_recommendations_shuffled():
    """Get all sleep recommendations in random order"""
    keys = []
    for key in r.scan_iter("sleep:*"):
        keys.append(key)

    if not keys:
        return []

    random.shuffle(keys)

    recommendations = []
    for key in keys:
        data = r.hgetall(key)
        rec_id = key.decode() if isinstance(key, bytes) else key
        rec_id = rec_id.split(":")[-1]

        recommendations.append({
            "id": rec_id,
            "title": data[b"title"].decode(),
            "brief_description": data[b"brief_description"].decode(),
            "detailed_description": data[b"detailed_description"].decode()
        })

    return recommendations


def find_song_by_title(title: str):
    """Find a song in Redis by its title"""
    for key in r.scan_iter("song:*"):
        song_data = r.hgetall(key)
        if song_data[b"title"].decode() == title:
            embedding_bytes = song_data[b"embedding"]
            embedding_array = np.frombuffer(embedding_bytes, dtype=np.float32)
            return {
                "title": song_data[b"title"].decode(),
                "embedding": embedding_array.tolist()
            }
    return None


def get_user_saved_recommendations(recommendation_ids: List[str]):
    """Get full recommendation data for saved recommendation IDs"""
    results = []

    for key in r.scan_iter("sleep:*"):
        data = r.hgetall(key)

        rec_id = key.decode() if isinstance(key, bytes) else key
        rec_id = rec_id.split(":")[-1]

        if rec_id not in recommendation_ids:
            continue

        embedding = None
        if b"embedding" in data:
            embedding = np.frombuffer(
                data[b"embedding"],
                dtype=np.float32
            ).tolist()

        results.append({
            "id": rec_id,
            "title": data[b"title"].decode(),
            "brief_description": data[b"brief_description"].decode(),
            "detailed_description": data[b"detailed_description"].decode(),
            "embedding": embedding
        })

    return results