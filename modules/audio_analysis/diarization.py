from pyannote.audio import Pipeline
from resemblyzer import VoiceEncoder, preprocess_wav
import numpy as np
import torch
import os
from collections import defaultdict
from pydub import AudioSegment
import json

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

try:
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization",
                                        use_auth_token="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")
except Exception as e:
    print(f"Failed to load pyannote pipeline: {e}")
    pipeline = None

encoder = VoiceEncoder()

def diarize_and_extract_speakers(audio_path, output_dir="output/speakers"):
    """
    Performs speaker diarization on the input audio and saves speaker-wise audio files and embeddings.

    Args:
        audio_path (str): Path to input audio (WAV).
        output_dir (str): Directory to save speaker audio files.

    Returns:
        Tuple[dict, str]:
            - speaker_data: dictionary of speaker segments, embeddings, and pause segments.
            - json_string: formatted JSON string of the speaker data.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[INFO] Running diarization on: {audio_path}")

    diarization = pipeline(audio_path)
    print("[INFO] Diarization complete.")

    audio = preprocess_wav(audio_path)
    audio_duration = len(audio) / encoder.sampling_rate
    print(f"[INFO] Audio duration: {round(audio_duration, 2)} seconds")

    segments_by_speaker = defaultdict(list)
    speaker_segments = defaultdict(list)
    all_segments = []

    print("[INFO] Parsing diarization results...")
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        start, end = round(turn.start, 2), round(turn.end, 2)
        print(f"[SPEAKER {speaker}] Start: {start}s, End: {end}s")
        all_segments.append((start, end, speaker))
        segments_by_speaker[speaker].append(audio[int(start * encoder.sampling_rate):int(end * encoder.sampling_rate)])
        speaker_segments[speaker].append({"start": start, "end": end})

    all_segments.sort()
    pause_segments = []
    print("[INFO] Detecting pause segments...")
    prev_end = 0.0
    for start, end, _ in all_segments:
        if start > prev_end:
            print(f"[PAUSE] From {round(prev_end, 2)}s to {round(start, 2)}s")
            pause_segments.append({"start": round(prev_end, 2), "end": round(start, 2)})
        prev_end = max(prev_end, end)

    if prev_end < audio_duration:
        pause_segments.append({"start": round(prev_end, 2), "end": round(audio_duration, 2)})
        print(f"[PAUSE] From {round(prev_end, 2)}s to {round(audio_duration, 2)}s (end of audio)")

    speaker_data = {}
    print("[INFO] Generating speaker embeddings and exporting audio...")
    for speaker, segments in segments_by_speaker.items():
        speaker_audio = np.concatenate(segments)
        embedding = encoder.embed_utterance(speaker_audio)

        speaker_id = f"speaker_{speaker}"
        file_path = os.path.join(output_dir, f"{speaker_id}.wav")
        AudioSegment(
            speaker_audio.tobytes(),
            frame_rate=encoder.sampling_rate,
            sample_width=2,
            channels=1
        ).export(file_path, format="wav")

        print(f"[SAVED] {file_path}")

        speaker_data[speaker_id] = {
            "segments": speaker_segments[speaker],
            "embedding": embedding.tolist(),
            "duration": round(len(speaker_audio) / encoder.sampling_rate, 2)
        }

    speaker_data["pause_segments"] = pause_segments

    print(f"[INFO] Processing complete. Found {len(speaker_data)-1} speakers and {len(pause_segments)} pause segments.\n")

    # Save as JSON
    json_path = os.path.join(output_dir, "speaker_embeddings.json")
    with open(json_path, "w") as f:
        json.dump(speaker_data, f, indent=2)

    print(f"[INFO] Speaker data saved to {json_path}")

    return speaker_data, json.dumps(speaker_data, indent=2)


#
# if __name__ == "__main__":
#     speakers = diarize_and_extract_speakers("full_audio.wav")
#     import json
#     with open("speaker_embeddings.json", "w") as f:
#         json.dump(speakers, f, indent=2)
