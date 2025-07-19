import asyncio
import edge_tts
import nest_asyncio
from pydub import AudioSegment
import os
import uuid

# Enable nested event loops (for notebooks or reentrant use)
nest_asyncio.apply()

# Emotion Profiles
emotion_profiles = {
    "neutral": {"gain": 0, "speed": 1.0},
    "happy": {"gain": 4, "speed": 1.1},
    "sad": {"gain": -3, "speed": 0.9},
}

def change_speed(sound, speed=1.0):
    sound_with_altered_frame_rate = sound._spawn(sound.raw_data, overrides={
        "frame_rate": int(sound.frame_rate * speed)
    })
    return sound_with_altered_frame_rate.set_frame_rate(sound.frame_rate)

def trim_silence(sound, silence_thresh=-40, padding=20):
    return sound.strip_silence(silence_len=100, silence_thresh=silence_thresh).fade_in(padding).fade_out(padding)

def compress_dynamic_range(sound):
    return sound.compress_dynamic_range()

# Async function to synthesize TTS using edge-tts and save as .mp3
async def synthesize_word(text, filename, voice="hi-IN-MadhurNeural"):
    try:
        communicate = edge_tts.Communicate(text=text, voice=voice)
        await communicate.save(filename)
    except Exception as e:
        print(f"❌ TTS Error for '{text}': {e}")

# Main function
def generate_emotional_speech(words_json, selected_emotion, target_language="hi", output_filename="final_emotion_output.wav"):
    if selected_emotion not in emotion_profiles:
        raise ValueError(f"Emotion '{selected_emotion}' is not defined in emotion profiles.")

    emotion = emotion_profiles[selected_emotion]
    output_audio = AudioSegment.silent(duration=0)
    temp_id = str(uuid.uuid4())[:8]

    for i, word in enumerate(words_json):
        text = word["text"]
        gain = word.get("gain", 0)
        speed = word.get("speed", 1.25)
        stress = word.get("stress", False)

        if stress:
            gain += 3
            speed *= 0.8

        total_gain = gain + emotion["gain"]
        total_speed = max(0.8, min(speed * emotion["speed"], 1.3))

        mp3_filename = f"temp_word_{i}_{temp_id}.mp3"

        # Generate MP3
        asyncio.run(synthesize_word(text, mp3_filename))

        if os.path.exists(mp3_filename):
            try:
                # Load MP3 and process
                audio = AudioSegment.from_file(mp3_filename, format="mp3")
                audio = audio.apply_gain(total_gain)
                audio = change_speed(audio, total_speed)
                audio = trim_silence(audio, silence_thresh=-40, padding=20)
                audio = audio.high_pass_filter(100)
                audio = compress_dynamic_range(audio)

                output_audio += audio
                os.remove(mp3_filename)
            except Exception as e:
                print(f"❌ Error processing audio for '{text}': {e}")
        else:
            print(f"❌ MP3 not generated for: {text}")

    # Normalize and export as WAV
    output_audio = output_audio.normalize()
    output_audio.export(output_filename, format="wav", codec="pcm_s16le")
    print(f"✅ Saved emotional speech as WAV → {output_filename}")
    return output_filename

# # Example input
# words_json = [
#     {"text": "नमस्ते", "stress": True},
#     {"text": "कैसे", "gain": 2},
#     {"text": "हैं", "speed": 0.6}
# ]
#
# # Call the function
# generate_emotional_speech(words_json, selected_emotion="happy", output_filename="output_emotion.wav")
