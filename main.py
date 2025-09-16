from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from fastapi.responses import FileResponse
import uuid
import os
import time
from app.routes.pipeline import text_text_translation, translate_and_generate_audio, text_text_translation_no_audio
from app.temp import process_video_with_subtitles
from typing import List, Dict, Optional
app = FastAPI()

# ---- CORS Settings ----
origins = [
    "http://localhost",
    "http://localhost:3000",  # frontend URL
    "http://127.0.0.1:3000",
    "*"  # allow all for testing (not recommended for production)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Middleware Example: Logging ----
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    print(f"{request.method} {request.url.path} completed in {process_time:.4f}s")
    return response

# ---- Request Models ----
class TextToTextRequest(BaseModel):
    source_text: str
    target_languages: str

class TextToSpeechRequest(BaseModel):
    source_text: str
    language: str


def text_to_speech(text: str, lang: str) -> str:
    audio_path = f"temp_audio_{uuid.uuid4().hex}.mp3"
    with open(audio_path, "wb") as f:
        f.write(b"FAKE AUDIO DATA")
    return audio_path

def speech_to_text(audio_path: str, lang: str) -> str:
    return "Detected text from audio"

def video_translate(video_path: str, target_lang: str) -> str:
    return f"Processed video for {target_lang}"

# ---- Endpoints ----

# Mapping full language names to ISO codes
LANG_MAP = {
    "hindi": "hi",
    "bengali": "bn",
    "tamil": "ta",
    "telugu": "te",
    "kannada": "kn",
    "malayalam": "ml",
    "marathi": "mr",
    "gujarati": "gu",
    "punjabi": "pa",
    "odia": "or",
    "assamese": "as",
    "urdu": "ur",
    "default": "df"
}

ALLOWED_CODES = set(LANG_MAP.values())  # {"hi","bn","ta",...}


@app.post("/text-to-speech")
async def text_to_text(req: TextToTextRequest):
    # Normalize input
    lang_input = req.target_languages.strip().lower()

    # Map full name → short code
    if lang_input in LANG_MAP:
        lang_code = LANG_MAP[lang_input]
    elif lang_input in ALLOWED_CODES:
        lang_code = lang_input
    else:
        lang_code = None

    if not lang_code:
        return [{
            "language": None,
            "translation": None,
            "audio_file": None
        }]

    results = await text_text_translation_no_audio(req.source_text, lang_code)
    return results

# @app.post("/text-to-speech")
# async def text_to_speech_endpoint(req: TextToSpeechRequest):
#     audio_path = text_to_speech(req.source_text, req.language)
#     return FileResponse(audio_path, media_type="audio/mpeg", filename="output.mp3")

@app.post("/speech-to-speech")
async def speech_to_speech_endpoint(
    file: UploadFile = File(...),
    language: str = Form(...)
):
    input_path = f"temp_input_{uuid.uuid4().hex}.wav"
    with open(input_path, "wb") as f:
        f.write(await file.read())
    text = speech_to_text(input_path, language)
    audio_path = text_to_speech(text, language)
    return FileResponse(audio_path, media_type="audio/mpeg", filename="output.mp3")

@app.post("/video-to-video")
async def video_to_video_endpoint(
    file: UploadFile = File(...),
    target_language: str = Form(...)
):
    # Save uploaded video temporarily
    temp_input_path = f"temp_upload_{uuid4().hex}.mp4"
    with open(temp_input_path, "wb") as f:
        f.write(await file.read())

    try:
        # Process video
        final_video_path = process_video_with_subtitles(temp_input_path, target_language)
        return FileResponse(final_video_path, media_type="video/mp4", filename="processed_video.mp4")
    finally:
        # Cleanup uploaded temp file
        if os.path.exists(temp_input_path):
            os.remove(temp_input_path)