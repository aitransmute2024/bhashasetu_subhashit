from pyannote.audio import Pipeline
from resemblyzer import VoiceEncoder, preprocess_wav
import numpy as np
import torch
import os
from collections import defaultdict
from pydub import AudioSegment
import json
from modules.audio_analysis.extract_pauses import pause_identification

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

try:
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization",
        use_auth_token="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"
    )
except Exception as e:
    print(f"Failed to load pyannote pipeline: {e}")
    pipeline = None

encoder = VoiceEncoder()
SAMPLING_RATE = 16000

# -------------------------
# Speaker Diarization
# -------------------------
def speaker_diarization(audio_path, output_dir):
    """
    Perform speaker diarization and return structured speaker data.
    """
    diarization = pipeline(audio_path)
    audio = preprocess_wav(audio_path)
    audio_duration = len(audio) / SAMPLING_RATE

    segments_by_speaker = defaultdict(list)
    speaker_segments = defaultdict(list)
    all_segments = []

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start, end = round(turn.start, 2), round(turn.end, 2)
        all_segments.append((start, end, speaker))
        segments_by_speaker[speaker].append(audio[int(start * SAMPLING_RATE):int(end * SAMPLING_RATE)])
        speaker_segments[speaker].append({"start": start, "end": end})

    all_segments.sort()

    speaker_data = {}
    for speaker, segments in segments_by_speaker.items():
        speaker_audio = np.concatenate(segments)
        embedding = encoder.embed_utterance(speaker_audio)

        speaker_id = f"speaker_{speaker}"
        file_path = os.path.join(output_dir, f"{speaker_id}.wav")
        AudioSegment(
            speaker_audio.tobytes(),
            frame_rate=SAMPLING_RATE,
            sample_width=2,
            channels=1
        ).export(file_path, format="wav")

        speaker_data[speaker_id] = {
            "segments": speaker_segments[speaker],
            "embedding": embedding.tolist(),
            "duration": round(len(speaker_audio) / SAMPLING_RATE, 2)
        }

    return speaker_data, all_segments, audio_duration


# -------------------------
# Main Function (unchanged name)
# -------------------------
def diarize_and_extract_speakers(audio_path, output_dir="output/speakers"):
    """
    Performs speaker diarization and pause identification, saves results to output.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n[INFO] Running diarization on: {audio_path}")
    speaker_data, all_segments, audio_duration = speaker_diarization(audio_path, output_dir)

    print("[INFO] Detecting pause segments...")
    pause_segments = pause_identification(all_segments, audio_duration)
    speaker_data["pause_segments"] = pause_segments

    json_path = os.path.join(output_dir, "speaker_embeddings.json")
    with open(json_path, "w") as f:
        json.dump(speaker_data, f, indent=2)

    print(f"[INFO] Processing complete. Found {len(speaker_data) - 1} speakers and {len(pause_segments)} pauses.")
    print(f"[INFO] Speaker data saved to {json_path}")

    return speaker_data
