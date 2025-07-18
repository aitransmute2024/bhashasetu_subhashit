import whisper
import parselmouth
import numpy as np
from parselmouth.praat import call
import json

def transcribe_words_with_timestamps(wav_path):
    model = whisper.load_model("base")  # or "small", "medium", etc.
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


def analyze_parselmouth_features_segment(snd, start_time, end_time):
    snd_segment = snd.extract_part(from_time=start_time, to_time=end_time, preserve_times=False)

    pitch = snd_segment.to_pitch()
    pitch_values = pitch.selected_array['frequency']
    mean_pitch = np.mean(pitch_values[pitch_values > 0]) if np.any(pitch_values > 0) else 0

    intensity = snd_segment.to_intensity()
    mean_intensity = call(intensity, "Get mean", 0, 0, "energy")

    formant = snd_segment.to_formant_burg()
    midpoint = snd_segment.get_total_duration() / 2
    f1 = call(formant, "Get value at time", 1, midpoint, 'Hertz', 'Linear')
    f2 = call(formant, "Get value at time", 2, midpoint, 'Hertz', 'Linear')
    f3 = call(formant, "Get value at time", 3, midpoint, 'Hertz', 'Linear')

    point_process = call(snd_segment, "To PointProcess (periodic, cc)", 75, 500)
    jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    shimmer = call([snd_segment, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

    harmonicity = snd_segment.to_harmonicity_cc()
    hnr = call(harmonicity, "Get mean", 0, 0)

    return {
        "Pitch": round(mean_pitch, 2),
        "Loudness": round(mean_intensity, 2),
        "Formant 1 (Hz)": round(f1, 2),
        "Formant 2 (Hz)": round(f2, 2),
        "Formant 3 (Hz)": round(f3, 2),
        "Jitter (local)": round(jitter, 4),
        "Shimmer (local)": round(shimmer, 4),
        "HNR (dB)": round(hnr, 2),
        "Duration (s)": round(snd_segment.get_total_duration(), 2)
    }


def extract_word_level_features(wav_path):
    """
    Extracts word-level acoustic features from a WAV file and returns results in JSON format.

    Args:
        wav_path (str): Path to the WAV audio file.

    Returns:
        str: JSON string containing list of word-level feature dictionaries.
    """
    snd = parselmouth.Sound(wav_path)
    word_timestamps = transcribe_words_with_timestamps(wav_path)  # assumed implemented elsewhere
    features = []

    for word_info in word_timestamps:
        word = word_info["word"]
        start = word_info["start"]
        end = word_info["end"]

        try:
            word_features = analyze_parselmouth_features_segment(snd, start, end)  # assumed implemented
            word_features.update({
                "word": word,
                "start": round(start, 2),
                "end": round(end, 2)
            })
            features.append(word_features)
        except Exception as e:
            print(f"⚠️ Error processing word '{word}': {e}")

    return json.dumps(features, indent=2)

# # 🔽 Usage Example
# if __name__ == "__main__":
#     wav_path = "output.wav"
#     word_level_data = extract_word_level_features(wav_path)
#
#     import json
#
#     print(json.dumps(word_level_data, indent=2))