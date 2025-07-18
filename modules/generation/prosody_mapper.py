import json


def map_target_to_prosodic_features(alignment_json, source_features_json, target_text):
    """
    Maps each target word to pitch and loudness using alignment and source word features.

    Parameters:
    - alignment_json: dict with key 'alignments', each having 'source', 'target'
    - source_features_json: list of dicts with 'word', 'pitch', 'loudness'
    - target_text: string (the complete target sentence)

    Returns:
    - List of dicts with 'target', 'pitch', 'loudness'
    """
    # Step 1: Split target sentence into words
    target_words = target_text.split()

    # Step 2: Map target word -> source word
    target_to_source = {entry["target"]: entry["source"] for entry in alignment_json.get("alignments", [])}

    # Step 3: Map source word -> {pitch, loudness}
    source_feature_map = {
        item["word"]: {"pitch": item["pitch"], "loudness": item["loudness"]}
        for item in source_features_json
    }

    # Step 4: Build final output
    result = []
    for target in target_words:
        source_word = target_to_source.get(target)
        feature = source_feature_map.get(source_word, {"pitch": 0, "loudness": 0})
        result.append({
            "word": target,
            "pitch_shift": feature["pitch"],
            "gain": feature["loudness"],
            "speed": 1.15
        })

    return result
