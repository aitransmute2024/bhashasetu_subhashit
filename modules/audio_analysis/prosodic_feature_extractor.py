import whisper
import parselmouth
import numpy as np
import json
from parselmouth.praat import call

# ----------------------------- Transcribe with Timestamps ----------------------------- #
def transcribe_words_with_timestamps(wav_path):
    model = whisper.load_model("base")  # Use "small", "medium", etc., for better accuracy
    result = model.transcribe(wav_path, word_timestamps=True, language='en')

    words_with_timestamps = []
    for segment in result["segments"]:
        for word_info in segment["words"]:
            words_with_timestamps.append({
                "word": word_info["word"].strip(),
                "start": word_info["start"],
                "end": word_info["end"]
            })
    return words_with_timestamps

# ----------------------------- Segment Words by Time ----------------------------- #
def segment_words_by_time(words_with_timestamps, max_duration=1.5):
    segments = []
    current_segment = []
    segment_start = None
    segment_end = None

    for word_info in words_with_timestamps:
        if not current_segment:
            segment_start = word_info['start']
            segment_end = word_info['end']
            current_segment.append(word_info)
        else:
            if word_info['end'] - segment_start <= max_duration:
                current_segment.append(word_info)
                segment_end = word_info['end']
            else:
                segments.append({
                    "words": current_segment,
                    "start": segment_start,
                    "end": segment_end
                })
                current_segment = [word_info]
                segment_start = word_info['start']
                segment_end = word_info['end']

    if current_segment:
        segments.append({
            "words": current_segment,
            "start": segment_start,
            "end": segment_end
        })

    return segments

# ----------------------------- Analyze Parselmouth Features ----------------------------- #
def analyze_parselmouth_features_segment(snd, start_time, end_time):
    snd_segment = snd.extract_part(from_time=start_time, to_time=end_time, preserve_times=False)

    pitch = snd_segment.to_pitch()
    pitch_values = pitch.selected_array['frequency']
    mean_pitch = np.mean(pitch_values[pitch_values > 0]) if np.any(pitch_values > 0) else 0

    intensity = snd_segment.to_intensity()
    mean_intensity = call(intensity, "Get mean", 0, 0, "energy")

    return {
        "pitch": round(mean_pitch, 2),
        "loudness": round(mean_intensity, 2)
    }

# ----------------------------- Compute Shifts ----------------------------- #
def compute_pitch_shift(pitch_value, base_pitch):
    if pitch_value <= 0 or base_pitch <= 0:
        return 0
    shift = 12 * np.log2(pitch_value / base_pitch)
    return int(round(shift))

def compute_loudness_shift(loudness_value, base_loudness):
    shift = (loudness_value - base_loudness) / 5.0
    return int(round(shift))

# ----------------------------- Extract Segment-Level Features ----------------------------- #
def extract_word_level_features(wav_path, max_duration=1.5):
    snd = parselmouth.Sound(wav_path)
    word_timestamps = transcribe_words_with_timestamps(wav_path)
    segments = segment_words_by_time(word_timestamps, max_duration=max_duration)

    features = []

    for segment in segments:
        start = segment['start']
        end = segment['end']
        words = [w['word'] for w in segment['words']]
        word_str = " ".join(words)

        try:
            acoustic_features = analyze_parselmouth_features_segment(snd, start, end)
            acoustic_features.update({
                "word": word_str,
                "start": round(start, 2),
                "end": round(end, 2)
            })
            features.append(acoustic_features)
        except Exception as e:
            print(f"⚠️ Error processing segment '{word_str}': {e}")

    # Compute baselines
    pitch_vals = [f["pitch"] for f in features if f["pitch"] > 0]
    loudness_vals = [f["loudness"] for f in features if f["loudness"] > 0]
    base_pitch = np.mean(pitch_vals) if pitch_vals else 200.0
    base_loudness = np.mean(loudness_vals) if loudness_vals else 70.0

    for f in features:
        f["pitch_shift"] = compute_pitch_shift(f["pitch"], base_pitch)
        f["loudness_shift"] = compute_loudness_shift(f["loudness"], base_loudness)

    # Save to JSON
    output_json = "output/segment_features.json"
    import os
    os.makedirs("output", exist_ok=True)
    with open(output_json, "w") as f_out:
        json.dump(features, f_out, indent=2)
    print(features)
    return features

# ----------------------------- Example Usage ----------------------------- #
# if __name__ == "__main__":
#     wav_path = "samples/sample.wav"  # 🔁 Change this to your WAV file path
#     segment_features = extract_segment_level_features(wav_path)
#
#     for seg in segment_features:
#         print(json.dumps(seg, indent=2))
