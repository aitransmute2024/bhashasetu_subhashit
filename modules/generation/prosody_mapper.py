import json
import string

def map_target_to_prosodic_features(alignment_json, source_features_json, target_text):
    target_words = [word.strip(string.punctuation).lower() for word in target_text.split()]

    # Map target to source
    target_to_source = {
        entry["target"].strip(string.punctuation).lower(): entry["source"].strip(string.punctuation).lower()
        for entry in alignment_json.get("alignments", [])
    }

    # Build source feature map safely
    source_feature_map = {}
    for i, item in enumerate(source_features_json):
        if isinstance(item, str):
            try:
                item = json.loads(item)
            except json.JSONDecodeError as e:
                # print(f"❌ Skipping invalid JSON at index {i}: {item}")
                continue
        if isinstance(item, dict) and "word" in item:
            key = item["word"].strip(string.punctuation).lower()
            source_feature_map[key] = {
                "pitch_shift": item.get("pitch_shift", 0),
                "loudness_shift": item.get("loudness_shift", 0)
            }

    # Final mapping
    result = []
    for target in target_words:
        source_word = target_to_source.get(target)
        feature = source_feature_map.get(source_word, {"pitch_shift": 0, "loudness_shift": 0})
        result.append({
            "text": target,
            "pitch_shift": feature["pitch_shift"],
            "gain": feature["loudness_shift"],
            "speed": 1.15
        })
    print(result)
    return result
