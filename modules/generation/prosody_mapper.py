import json
import string

def map_target_to_prosodic_features(alignment_json, source_features_json, target_text, duration_threshold=0.2):
    target_words = [word.strip(string.punctuation).lower() for word in target_text.split()]

    # Map target to source
    target_to_source = {
        entry["target"].strip(string.punctuation).lower(): entry["source"].strip(string.punctuation).lower()
        for entry in alignment_json.get("alignments", [])
    }

    # Build source feature map
    source_feature_map = {}
    for i, item in enumerate(source_features_json):
        if isinstance(item, str):
            try:
                item = json.loads(item)
            except json.JSONDecodeError:
                continue  # skip malformed entries

        if isinstance(item, dict) and "word" in item:
            key = item["word"].strip(string.punctuation).lower()
            start = item.get("start", 0)
            end = item.get("end", 0)
            duration = max(end - start, 0)
            source_feature_map[key] = {
                "pitch_shift": item.get("pitch_shift", 0),
                "loudness_shift": item.get("loudness_shift", 0),
                "start": start,
                "end": end,
                "duration": duration
            }

    # Final mapping
    result = []
    for target in target_words:
        source_word = target_to_source.get(target)
        feature = source_feature_map.get(source_word, {
            "pitch_shift": 0,
            "loudness_shift": 0,
            "start": 0,
            "end": 0,
            "duration": 0
        })

        is_stressed = feature["duration"] > duration_threshold

        result.append({
            "text": target,
            "pitch_shift": feature["pitch_shift"],
            "gain": feature["loudness_shift"],
            "speed": 1,
            "stress": is_stressed
        })

    return result
