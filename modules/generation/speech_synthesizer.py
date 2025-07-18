from gtts import gTTS
from pydub import AudioSegment
import librosa
import soundfile as sf
import os
import numpy as np
import uuid

# Define emotion profiles globally
emotion_profiles = {
    "neutral": {"pitch_shift": 0, "gain": 0, "speed": 1.5},
    "happy": {"pitch_shift": 3, "gain": 4, "speed": 1.7},
    "sad": {"pitch_shift": -2, "gain": -3, "speed": 1.2},
    "angry": {"pitch_shift": 4, "gain": 6, "speed": 1.8},
    "fear": {"pitch_shift": 2, "gain": 2, "speed": 1.6},
    "calm": {"pitch_shift": -1, "gain": -1, "speed": 1.3},
    "disgust": {"gain": -3, "pitch_shift": -1, "speed": 1.25},
    "surprise": {"gain": 4, "pitch_shift": 3, "speed": 1.15}

}

def generate_emotional_speech(words_json, selected_emotion, target_language,output_filename="final_emotion_output.wav"):
    if selected_emotion not in emotion_profiles:
        raise ValueError(f"Emotion '{selected_emotion}' is not defined in the emotion profiles.")

    emotion = emotion_profiles[selected_emotion]
    output_audio = AudioSegment.empty()
    temp_ids = str(uuid.uuid4())[:8]  # Unique suffix to avoid filename collision

    for i, word in enumerate(words_json):
        text = word["text"]
        gain = word.get("gain", 0)
        pitch_shift = word.get("pitch_shift", 0)
        speed = word.get("speed", 1.0)
        stress = word.get("stress", False)

        if stress:
            gain += 3
            pitch_shift += 1
            speed *= 0.8  # slower for emphasis

        total_gain = gain + emotion["gain"]
        total_pitch = pitch_shift + emotion["pitch_shift"]
        total_speed = speed * emotion["speed"]

        # Generate TTS and save
        tts = gTTS(text=text, lang=target_language)
        tts_filename = f"word_{i}_{temp_ids}.mp3"
        wav_filename = f"word_{i}_{temp_ids}.wav"
        shifted_filename = f"shifted_{i}_{temp_ids}.wav"

        tts.save(tts_filename)

        # Convert to WAV
        AudioSegment.from_mp3(tts_filename).export(wav_filename, format="wav")

        # Apply pitch shift
        y, sr = librosa.load(wav_filename, sr=None)
        y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=total_pitch)
        sf.write(shifted_filename, y_shifted, sr)

        # Apply gain and speed
        audio = AudioSegment.from_wav(shifted_filename)
        audio = audio.apply_gain(total_gain)
        new_frame_rate = int(audio.frame_rate * total_speed)
        audio = audio._spawn(audio.raw_data, overrides={"frame_rate": new_frame_rate})
        audio = audio.set_frame_rate(44100)

        output_audio += audio

        # Cleanup
        for file in [tts_filename, wav_filename, shifted_filename]:
            if os.path.exists(file):
                os.remove(file)

    output_audio.export(output_filename, format="wav")
    print(f"✅ Saved audio with emotion '{selected_emotion}' and per-word modifiers → {output_filename}")
    return output_filename
