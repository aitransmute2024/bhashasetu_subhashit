from gtts import gTTS
from pydub import AudioSegment, silence
import os
import uuid

# Emotion profile definitions
emotion_profiles = {
    "neutral": {"gain": 0, "speed": 1.2},
    "happy": {"gain": 4, "speed": 1.1},
    "sad": {"gain": -3, "speed": 0.95},
    "angry": {"gain": 6, "speed": 1.3},
    "fear": {"gain": 2, "speed": 1.2},
    "calm": {"gain": -1, "speed": 1},
    "disgust": {"gain": -3, "speed": 1.05},
    "surprise": {"gain": 4, "speed": 1.05}
}

def change_speed(audio, speed=1.0):
    """Adjust playback speed (affects pitch) by changing frame rate."""
    new_frame_rate = int(audio.frame_rate * speed)
    return audio._spawn(audio.raw_data, overrides={'frame_rate': new_frame_rate}).set_frame_rate(44100)

def trim_silence(audio, silence_thresh=-40, padding=20):
    """Trim leading and trailing silence from an AudioSegment."""
    start_trim = silence.detect_leading_silence(audio, silence_thresh)
    end_trim = silence.detect_leading_silence(audio.reverse(), silence_thresh)
    duration = len(audio)
    trimmed = audio[start_trim:duration - end_trim]
    return AudioSegment.silent(duration=padding) + trimmed + AudioSegment.silent(duration=padding)

def generate_emotional_speech(words_json, selected_emotion, target_language, output_filename="final_emotion_output.wav"):
    if selected_emotion not in emotion_profiles:
        raise ValueError(f"Emotion '{selected_emotion}' is not defined in the emotion profiles.")

    emotion = emotion_profiles[selected_emotion]
    output_audio = AudioSegment.silent(duration=0)
    temp_ids = str(uuid.uuid4())[:8]

    for i, word in enumerate(words_json):
        text = word["text"]
        gain = word.get("gain", 0)
        speed = word.get("speed", 1.25)
        stress = word.get("stress", False)

        if stress:
            gain += 3
            speed *= 0.8  # slower for emphasis

        total_gain = gain + emotion["gain"]
        total_speed = speed * emotion["speed"]

        # Generate TTS audio
        tts = gTTS(text=text, lang=target_language)
        tts_filename = f"word_{i}_{temp_ids}.mp3"
        tts.save(tts_filename)

        # Process audio
        audio = AudioSegment.from_file(tts_filename, format="mp3")
        audio = audio.apply_gain(total_gain)
        audio = change_speed(audio, total_speed)
        audio = trim_silence(audio, silence_thresh=-40, padding=20)

        output_audio += audio

        os.remove(tts_filename)

    output_audio.export(output_filename, format="wav")
    print(f"✅ Saved audio with emotion '{selected_emotion}' → {output_filename}")
    return output_filename
