from utils.speaker_features import extract_voice_features
import os
import json
# ---------------------------
# Create Database from Available Voices
# ---------------------------
def create_voice_database(voice_dir, output_json="speaker_features.json"):
    voices_json = {}
    for fname in os.listdir(voice_dir):
        if fname.lower().endswith(".wav"):
            path = os.path.join(voice_dir, fname)
            print(f"Processing {fname}...")
            features = extract_voice_features(path)
            voices_json[fname] = features
    with open(output_json, "w") as f:
        json.dump(voices_json, f, indent=2)
    print(f"Saved to {output_json}")
