from deepface import DeepFace
import os
import json
import glob
import numpy as np

def analyze_faces(image_path):
    try:
        results = DeepFace.analyze(
            img_path=image_path,
            actions=["gender", "emotion"],
            enforce_detection=False
        )
        if isinstance(results, list):
            return results
        else:
            return [results]
    except Exception as e:
        print(f"[ERROR] Failed to analyze {image_path}: {e}")
        return []

def load_captions(caption_file="captions.json"):
    if os.path.exists(caption_file):
        with open(caption_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def build_scene_metadata(image_folder="scene_frames", output_file="scene_metadata.json", caption_file="captions.json"):
    image_paths = sorted(glob.glob(os.path.join(image_folder, "scene_*.jpg")))
    captions = load_captions(caption_file)
    all_metadata = {}

    for img_path in image_paths:
        scene_id = os.path.basename(img_path)
        print(f"[INFO] Analyzing {scene_id}")
        faces = analyze_faces(img_path)

        speakers = []
        for face in faces:
            speakers.append({
                "gender": face.get("gender", "unknown"),
                "emotion": face.get("dominant_emotion", "neutral")
            })

        all_metadata[scene_id] = {
            "caption": captions.get(scene_id, ""),
            "num_speakers": len(speakers),
            "speakers": speakers
        }


    def make_json_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_json_serializable(v) for v in obj]
        elif isinstance(obj, np.generic):
            return obj.item()
        else:
            return obj

    with open("scene_metadata.json", "w", encoding="utf-8") as f:
        json.dump(make_json_serializable(all_metadata), f, indent=2, ensure_ascii=False)
    print(f"[✓] Metadata saved to {output_file}")

if __name__ == "__main__":
    build_scene_metadata()
