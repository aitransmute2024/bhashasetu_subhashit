import os
import numpy as np
import parselmouth
from parselmouth.praat import call
from pydub import AudioSegment
import librosa
import webrtcvad
import collections
import contextlib
import wave
# import whisper
# from speechbrain.pretrained import SpeakerRecognition


def convert_mp3_to_wav(mp3_path, wav_path):
    audio = AudioSegment.from_mp3(mp3_path)
    audio.export(wav_path, format="wav")


def analyze_parselmouth_features(wav_path):
    snd = parselmouth.Sound(wav_path)
    pitch = snd.to_pitch()
    pitch_values = pitch.selected_array['frequency']
    mean_pitch = np.mean(pitch_values[pitch_values > 0])

    intensity = snd.to_intensity()
    mean_intensity = call(intensity, "Get mean", 0, 0, "energy")

    formant = snd.to_formant_burg()
    midpoint = snd.get_total_duration() / 2
    f1 = call(formant, "Get value at time", 1, midpoint, 'Hertz', 'Linear')
    f2 = call(formant, "Get value at time", 2, midpoint, 'Hertz', 'Linear')
    f3 = call(formant, "Get value at time", 3, midpoint, 'Hertz', 'Linear')

    point_process = call(snd, "To PointProcess (periodic, cc)", 75, 500)
    jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    shimmer = call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)

    harmonicity = snd.to_harmonicity_cc()
    hnr = call(harmonicity, "Get mean", 0, 0)

    return {
        "Pitch (Hz)": round(mean_pitch, 2),
        "Mean Intensity (dB)": round(mean_intensity, 2),
        "Formant 1 (Hz)": round(f1, 2),
        "Formant 2 (Hz)": round(f2, 2),
        "Formant 3 (Hz)": round(f3, 2),
        "Jitter (local)": round(jitter, 4),
        "Shimmer (local)": round(shimmer, 4),
        "HNR (dB)": round(hnr, 2),
        "Duration (s)": round(snd.get_total_duration(), 2)
    }


def extract_librosa_features(wav_path):
    y, sr = librosa.load(wav_path)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = mfccs.mean(axis=1)

    centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
    flatness = librosa.feature.spectral_flatness(y=y).mean()
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()

    return {
        "MFCCs": [round(val, 2) for val in mfccs_mean],
        "Spectral Centroid": round(centroid, 2),
        "Spectral Bandwidth": round(bandwidth, 2),
        "Spectral Flatness": round(flatness, 4),
        "Spectral Rolloff": round(rolloff, 2)
    }


from pyAudioAnalysis import audioTrainTest as aT

def extract_emotion(wav_path):
    # Pre-trained model path (use the one bundled with pyAudioAnalysis or your own)
    model_path = "pyAudioAnalysis/data/svmModel"  # replace if using a custom model
    model_type = "svm"

    try:
        # returns: class label, class probability, class names
        result, prob, classes = aT.classifyFileWrapper(wav_path, model_path, model_type)
        return {
            "Predicted Emotion": result,
            "Emotion Probabilities": dict(zip(classes, map(lambda x: round(x, 3), prob)))
        }
    except Exception as e:
        return {"Emotion Detection Error": str(e)}

# def get_transcription_and_speech_rate(wav_path):
#     model = whisper.load_model("base")
#     result = model.transcribe(wav_path)
#     text = result["text"]
#     duration = librosa.get_duration(filename=wav_path)
#     words = text.strip().split()
#     rate = len(words) / duration if duration > 0 else 0
#     return {
#         "Transcription": text.strip(),
#         "Speech Rate (words/sec)": round(rate, 2)
#     }
#

# def extract_speaker_embedding(wav_path):
#     classifier = SpeakerRecognition.from_hparams(
#         source="speechbrain/spkrec-ecapa-voxceleb",
#         savedir="tmp/spkrec"
#     )
#     embedding = classifier.encode_file(wav_path)
#     return {"Speaker Embedding": embedding.squeeze().tolist()}


def analyze_voice_extended(wav_path):
    features = {}
    features.update(analyze_parselmouth_features(wav_path))
    features.update(extract_librosa_features(wav_path))
    features.update(extract_emotion(wav_path))  # Add emotion here
    return features

# 🔽 Usage Example
if __name__ == "__main__":
    mp3_file = "sample.wav"
    results = analyze_voice_extended(mp3_file)

    for k, v in results.items():
        if isinstance(v, list):
            print(f"{k}: {', '.join(map(str, v))}")
        else:
            print(f"{k}: {v}")
