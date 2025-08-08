import os
import json
import numpy as np
import librosa
import soundfile as sf
import parselmouth  # Praat
from speechbrain import EncoderClassifier

# Load pretrained speaker embedding model once
speaker_model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

# ---------------------------
# Feature Extraction Functions
# ---------------------------

def extract_formants_praat(y, sr):
    snd = parselmouth.Sound(y, sr)
    formant = snd.to_formant_burg()
    formants = []
    for i in range(1, 4):  # first 3 formants
        try:
            f = [formant.get_value_at_time(i, t) for t in np.linspace(0, snd.duration, 5)]
            f = [val for val in f if val is not None]
            formants.append(np.mean(f) if f else 0)
        except:
            formants.append(0)
    return formants

def extract_jitter_shimmer(y, sr):
    snd = parselmouth.Sound(y, sr)
    pt = snd.to_point_process_cc()
    jitter = parselmouth.praat.call(pt, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    shimmer = parselmouth.praat.call([snd, pt], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    return jitter, shimmer

def extract_embedding(wav_path):
    signal, fs = librosa.load(wav_path, sr=16000)  # Speechbrain prefers 16kHz
    embedding = speaker_model.encode_batch(torch.tensor(signal).unsqueeze(0))
    return embedding.squeeze().detach().cpu().numpy()

def extract_voice_features(wav_path):
    y, sr = librosa.load(wav_path, sr=None)

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)

    # Pitch
    pitch, mag = librosa.piptrack(y=y, sr=sr)
    pitches = pitch[pitch > 0]
    avg_pitch = np.mean(pitches) if len(pitches) > 0 else 0

    # RMS Energy
    rms = np.mean(librosa.feature.rms(y=y))

    # Spectral Features
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))

    # Zero Crossing Rate
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))

    # Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    # Duration
    duration = librosa.get_duration(y=y, sr=sr)

    # Formants
    formants = extract_formants_praat(y, sr)

    # Jitter/Shimmer
    jitter, shimmer = extract_jitter_shimmer(y, sr)

    # Speaker Embedding
    embedding = extract_embedding(wav_path).tolist()

    return {
        "mfcc": mfcc.tolist(),
        "avg_pitch": float(avg_pitch),
        "rms": float(rms),
        "spectral_centroid": float(spectral_centroid),
        "spectral_bandwidth": float(spectral_bandwidth),
        "spectral_contrast": float(spectral_contrast),
        "spectral_rolloff": float(spectral_rolloff),
        "zero_crossing_rate": float(zcr),
        "tempo": float(tempo),
        "duration": float(duration),
        "formant1": float(formants[0]),
        "formant2": float(formants[1]),
        "formant3": float(formants[2]),
        "jitter": float(jitter),
        "shimmer": float(shimmer),
        "embedding": embedding
    }



