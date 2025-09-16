# 🎬 SuBhashit: AI-Powered Cross-Lingual Video Translator

SuBhashit is an advanced multilingual video translation pipeline that transforms spoken content from one language to another, preserving the **emotion**, **context**, **prosody**, and **speaker style** of the original video. Designed for scale, modularity, and cultural relevance, SuBhashit provides a seamless backend interface for end-to-end video localization.

> **Built with Python 3.12 | FastAPI | PyAnnote | RVC | IndicTTS | Facebook NLLB | Whisper | Transformers**

---

## ✨ Features

- 🎯 Automatic video scene segmentation and shot detection
- 🧠 Speaker diarization, gender/age/accent/emotion recognition
- ✍️ High-quality ASR and emotion-aware neural translation
- 🗣️ Prosody-aligned text-to-speech with voice cloning
- 🕵️‍♀️ Lip-sync compatible voice synthesis (optional)
- 💬 Auto subtitle generation and export/burn-in options
- 🧩 Fully modular, extendable, and testable backend

---

## 🛠️ Pipeline Architecture

### 🔹 1. Preprocessing
- **Video Ingestion** – Load video, validate resolution/framerate
- **Scene & Shot Detection** – Segment into logical scenes and fine-grained shots

### 🔹 2. Audio Processing
- **Audio Extraction & Normalization** – Isolate and standardize audio
- **Noise Reduction** – Enhance clarity and reduce background noise
- **Speaker Diarization** – PyAnnote-based multi-speaker separation
- **Demographic & Emotion Analysis** – Detect gender, age, accent, emotion

### 🔹 3. Speech-to-Text & NLP
- **ASR** – Transcribe speech (Whisper)
- **Punctuation & Cleanup** – Format output and remove disfluencies
- **NER & Idiom Detection** – Detect entities and culturally-bound phrases
- **Text Emotion Analysis** – Emotional tone from transcribed content

### 🔹 4. Translation & Localization
- **Translation** – Neural machine translation via Facebook NLLB-200 distilled
- **Cultural Adaptation** – Idiom/context/localization adaptation
- **Keyword Tagging** – Highlight key terms, maintain translation fidelity

### 🔹 5. Voice Synthesis
- **Prosody Mapping** – Align prosody to translated text
- **TTS** – AI4Bharat IndicParlerTTS + RVC voice cloning
- **Duration Control** – Adjust speech length, stretch/compress timing

### 🔹 6. Video Reconstruction
- **Audio Sync & Lip Alignment** – Sync audio with visual cues
- **Subtitles** – Auto-generate styled SRT or hardcoded subtitles

### 🔹 7. Postprocessing
- **Encoding** – Compress final output for deployment
- **QA & Export** – Final validation and delivery

---

## 📁 Project Structure

```
project_root/
├── app/                      # FastAPI Application Layer
│   ├── main.py
│   ├── routes/
│   │   ├── pipeline.py
│   │   ├── preprocess.py
│   │   ├── generation.py
│   │   ├── text_analysis.py
│   │   └── voice_analysis.py
│   └── dependencies.py
│
├── core/                     # Pipeline Orchestration
│   ├── config.py
│   └── pipeline_controller.py
│
├── modules/
│   ├── preprocessing/
│   ├── voice_analysis/
│   ├── text_analysis/
│   └── generation/
│
├── models/                   # Pretrained Model Weights
│   ├── voice/
│   └── text/
│
├── utils/                    # Logging, File & Audio Tools
├── data/                     # Input/Processed/Output Media
├── tests/                    # Unit & Integration Tests
├── requirements.txt
└── README.md
```

---

## 📦 Dependencies

**Core Libraries & Frameworks:**
- `FastAPI`, `uvicorn`, `pydantic`, `ffmpeg-python`
- `PyAnnote` – speaker diarization
- `whisper`, `transformers`, `sentencepiece`
- `facebook/nllb-200-distilled-600M` – translation
- `ai4bharat/IndicParlerTTS` – TTS for Indian languages
- `RVC` – retrieval-based voice cloning
- `librosa`, `torchaudio`, `scipy`, `numpy`, `spaCy`

---

## 🚀 Getting Started

### 🔧 Setup

```bash
git clone https://github.com/your-org/subhashit.git
cd subhashit

python3.12 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### ▶️ Run the Server

```bash
uvicorn app.main:app --reload
```

### 📬 API Example

#### POST `/pipeline/run`
```json
{
  "input_video_path": "data/input/sample.mp4",
  "target_language": "hi"
}
```

Returns: Translated video path, subtitle file

---

## 🧪 Testing

```bash
pytest tests/
```

---

## 📤 Output

- 🎞️ Translated Video (mp4)
- 💬 Subtitles (SRT or burned-in)
- 🗂️ Logs and intermediate files in `/data/processed`

---

## 🤝 Contributing

We welcome issues, improvements, and new module integrations.
Please submit a pull request or open a discussion.

---

## 📝 License

MIT License © 2025 AiTransmute 

---

## 📚 Acknowledgements

- [AI4Bharat](https://ai4bharat.org)
- [Facebook NLLB](https://ai.facebook.com/research/no-language-left-behind)
- [OpenAI Whisper](https://github.com/openai/whisper)
- [RVC Voice Conversion](https://github.com/RVC-Project)
- [PyAnnote Speaker Diarization](https://github.com/pyannote/pyannote-audio)