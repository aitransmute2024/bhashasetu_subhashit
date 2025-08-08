import json
import numpy as np
from scipy.spatial.distance import cosine
from utils.speaker_features import extract_voice_features

def load_voice_database(json_path="voices_features.json"):
    with open(json_path, "r") as f:
        return json.load(f)

def feature_vector_from_dict(feat_dict):
    # Combine all features except embeddings, which we treat separately
    non_embed_features = (
        feat_dict["mfcc"] +
        [
            feat_dict["avg_pitch"], feat_dict["rms"],
            feat_dict["spectral_centroid"], feat_dict["spectral_bandwidth"],
            feat_dict["spectral_contrast"], feat_dict["spectral_rolloff"],
            feat_dict["zero_crossing_rate"], feat_dict["tempo"],
            feat_dict["duration"], feat_dict["formant1"],
            feat_dict["formant2"], feat_dict["formant3"],
            feat_dict["jitter"], feat_dict["shimmer"]
        ]
    )
    return np.array(non_embed_features), np.array(feat_dict["embedding"])

def find_closest_voice(new_wav, database):
    new_features = extract_voice_features(new_wav)
    new_vec, new_embed = feature_vector_from_dict(new_features)

    best_match = None
    best_score = float("inf")

    for fname, feats in database.items():
        vec, embed = feature_vector_from_dict(feats)

        # Combine distances: spectral+prosodic features (Euclidean) and embedding (cosine)
        dist_features = np.linalg.norm(new_vec - vec)
        dist_embed = cosine(new_embed, embed)

        total_score = 0.5 * dist_features + 0.5 * dist_embed  # weighting

        if total_score < best_score:
            best_score = total_score
            best_match = fname

    return best_match, best_score


# ---------------------------
# Usage Example
# ---------------------------
# if __name__ == "__main__":
#     # STEP 1: Build database from available voices
#     # create_voice_database("./available_voices")
#
#     # STEP 2: Load database and find match for a new voice sample
#     db = load_voice_database("voices_features.json")
#     closest, score = find_closest_voice("new_voice.wav", db)
#     print(f"Closest match: {closest} (Score: {score:.4f})")