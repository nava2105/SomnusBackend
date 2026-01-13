import os
from datetime import datetime, timedelta
from jose import jwt
from dotenv import load_dotenv
import bcrypt

load_dotenv()

JWT_SECRET = os.getenv("JWT_SECRET", "CHANGE_ME_32CHARS_MIN")
JWT_ALG = "HS256"
JWT_EXPIRE_MINUTES = 60 * 24

def hash_password(password: str) -> str:
    # bcrypt accepts only bytes
    salt = bcrypt.gensalt(rounds=12)
    hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
    return hashed.decode("utf-8")  # save string in Mongo

def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(
        password.encode("utf-8"),
        password_hash.encode("utf-8")
    )

def create_token(subject: str) -> str:
    expire = datetime.utcnow() + timedelta(minutes=JWT_EXPIRE_MINUTES)
    payload = {"sub": subject, "exp": expire}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)
