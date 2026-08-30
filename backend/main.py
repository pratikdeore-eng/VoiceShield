from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import io
import numpy as np
import soundfile as sf
import torch
import av
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

app = FastAPI(title="VoiceGuard API")

# CORS - allow frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ================= AUTHENTICATION =================

import sqlite3
import hashlib
import os


DATABASE = "voiceguard.db"


def get_db():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    hashed = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        100000
    )

    return salt.hex() + ":" + hashed.hex()


def verify_password(password: str, stored_password: str) -> bool:
    try:
        salt_hex, hash_hex = stored_password.split(":")
        salt = bytes.fromhex(salt_hex)

        hashed = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            100000
        )

        return hashed.hex() == hash_hex

    except Exception:
        return False


def init_database():

    conn = get_db()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


init_database()


class RegisterRequest(BaseModel):
    name: str
    email: str
    password: str

users = {}
class LoginRequest(BaseModel):
    email: str
    password: str
class RegisterRequest(BaseModel):
    name: str
    email: str
    password: str

@app.post("/api/register")
async def register_user(data: RegisterRequest):

    name = data.name.strip()
    email = data.email.strip().lower()
    password = data.password

    if not name or not email or not password:
        raise HTTPException(
            status_code=400,
            detail="All fields are required"
        )

    if len(password) < 6:
        raise HTTPException(
            status_code=400,
            detail="Password must be at least 6 characters"
        )

    conn = get_db()

    existing_user = conn.execute(
        "SELECT id FROM users WHERE email = ?",
        (email,)
    ).fetchone()

    if existing_user:
        conn.close()

        raise HTTPException(
            status_code=409,
            detail="Email already registered"
        )

    hashed_password = hash_password(password)

    conn.execute(
        """
        INSERT INTO users (name, email, password)
        VALUES (?, ?, ?)
        """,
        (name, email, hashed_password)
    )

    conn.commit()
    conn.close()

    return {
        "status": "success",
        "message": "Registration successful",
        "user": {
            "name": name,
            "email": email
        }
    }


@app.post("/api/login")
async def login_user(data: LoginRequest):

    email = data.email.strip().lower()
    password = data.password

    conn = get_db()

    user = conn.execute(
        """
        SELECT id, name, email, password
        FROM users
        WHERE email = ?
        """,
        (email,)
    ).fetchone()

    conn.close()

    if not user:
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password"
        )

    if not verify_password(password, user["password"]):
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password"
        )

    return {
        "status": "success",
        "message": "Login successful",
        "user": {
            "name": user["name"],
            "email": user["email"]
        }
    }
MODEL_NAME = "garystafford/wav2vec2-deepfake-voice-detector"

feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
model = AutoModelForAudioClassification.from_pretrained(MODEL_NAME)

model.eval()
def detect_voice_type(audio, sample_rate):
    # Convert audio to 16 kHz mono for the model
    audio = np.asarray(audio, dtype=np.float32)

    if sample_rate != 16000:
        import librosa
        audio = librosa.resample(
            audio,
            orig_sr=sample_rate,
            target_sr=16000
        )
        sample_rate = 16000

    chunk_size = 16000 * 10  # 10 seconds
    predictions = []

    for start in range(0, len(audio), chunk_size):
        chunk = audio[start:start + chunk_size]

        # Ignore very short final chunks
        if len(chunk) < 16000 * 2:
            continue

        inputs = feature_extractor(
            chunk,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )

        with torch.no_grad():
            outputs = model(**inputs)

        print("MODEL LABELS:", model.config.id2label)
        print("MODEL LOGITS:", outputs.logits)
        
        probabilities = torch.softmax(outputs.logits, dim=-1)[0]
        predictions.append(probabilities.numpy())

    if not predictions:
        raise ValueError("Audio is too short for AI detection")

    average_prediction = np.mean(predictions, axis=0)

    real_probability = float(average_prediction[0])
    ai_probability = float(average_prediction[1])

    if ai_probability >= real_probability:
        verdict = "AI-GENERATED"
        confidence = ai_probability
    else:
        verdict = "HUMAN"
        confidence = real_probability

    return verdict, confidence
@app.get("/")
def home():
    return {
        "message": "VoiceGuard Backend is running!",
        "status": "success"
    }

def home():
    return {
        "message": "VoiceGuard Backend is running!",
        "status": "success"
    }


@app.post("/api/analyze")
async def analyze_audio(file: UploadFile = File(...)):

    allowed_types = [
        "audio/wav",
        "audio/x-wav",
        "audio/wave",
        "audio/mpeg",
        "audio/mp3",
        "audio/webm",
        "audio/ogg",
        "audio/opus",
        "video/mpeg",
    ]

    if not (
        file.content_type in allowed_types
        or file.content_type.startswith("audio/webm")
    ):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid audio format: {file.content_type}"
        )

    audio_data = await file.read()

    if len(audio_data) == 0:
        raise HTTPException(
            status_code=400,
            detail="Empty audio file"
        )

    try:

        # Decode WebM/Opus audio from browser
        container = av.open(io.BytesIO(audio_data))

        audio_stream = container.streams.audio[0]
        sample_rate = audio_stream.rate

        frames = []

        for frame in container.decode(audio=0):

            frame_array = frame.to_ndarray()

            # Convert multi-channel audio to mono
            if frame_array.ndim > 1:
                frame_array = np.mean(
                    frame_array,
                    axis=0
                )

            frames.append(frame_array)

        if not frames:
            raise ValueError("No audio frames found")

        audio = np.concatenate(frames)

        # Convert audio to float32
        audio = audio.astype(np.float32)

        # Normalize if necessary
        max_value = np.max(np.abs(audio))

        if max_value > 1:
            audio = audio / max_value

        # Convert to 16 kHz for AI model
        if sample_rate != 16000:

            import librosa

            audio = librosa.resample(
                audio,
                orig_sr=sample_rate,
                target_sr=16000
            )

            sample_rate = 16000

        duration = len(audio) / sample_rate

        rms = float(
            np.sqrt(np.mean(audio ** 2))
        )

        zero_crossings = np.sum(
            np.abs(np.diff(np.sign(audio))) > 0
        )

        zero_crossing_rate = float(
            zero_crossings / len(audio)
        )

        peak_amplitude = float(
            np.max(np.abs(audio))
        )

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=f"Could not process audio: {str(e)}"
        )

    verdict, confidence = detect_voice_type(
        audio,
        sample_rate
    )

    return {
        "filename": file.filename,
        "status": "processed",
        "message": "Audio processed successfully",

        "audio_info": {
            "size_bytes": len(audio_data),
            "content_type": file.content_type,
            "sample_rate": sample_rate,
            "duration_seconds": round(duration, 2)
        },

        "features": {
            "rms": round(rms, 6),
            "zero_crossing_rate": round(
                zero_crossing_rate,
                6
            ),
            "peak_amplitude": round(
                peak_amplitude,
                6
            )
        },

        "ai_detection": {
            "verdict": verdict,
            "confidence": round(confidence, 4)
        }
    }