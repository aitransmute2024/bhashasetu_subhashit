from scenedetect import detect, ContentDetector
import json
import os


def extract_scenes(video_path, output_json='output/scene_timestamps.json', threshold=15.0):
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    # Detect scenes using content-based detection
    scene_list = detect(video_path, ContentDetector(threshold=threshold))

    # Convert scene start/end times to seconds and prepare JSON
    scene_timestamps = [
        {
            "scene": f"scene{i + 1}",
            "start_time": start.get_seconds(),
            "end_time": end.get_seconds()
        }
        for i, (start, end) in enumerate(scene_list)
    ]

    # Save scene timestamps as JSON
    with open(output_json, 'w') as f:
        json.dump(scene_timestamps, f, indent=2)

    return output_j
    son  # Return path to JSON for chaining or logging
