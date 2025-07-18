from pydub import AudioSegment
import json
import os

def split_audio_by_scenes(full_audio_path, scene_json, output_dir="output/scenes_audio"):
    """
    Splits the full audio into scene-based chunks using timestamps from the JSON.

    Args:
        full_audio_path (str): Path to the full audio file.
        scene_json (str): Path to the JSON file containing scene timestamps.
        output_dir (str): Directory to save the scene-based audio chunks.

    Returns:
        List[str]: List of output audio file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    audio = AudioSegment.from_wav(full_audio_path)
    output_files = []

    with open(scene_json, 'r') as f:
        try:
            scenes = json.load(f)
        except json.JSONDecodeError:
            scenes = []

    if not scenes:
        # No scenes — save full audio as one file
        output_path = os.path.join(output_dir, "scene1.wav")
        audio.export(output_path, format="wav")
        print(f"🔁 No scenes found — saved full audio as: {output_path}")
        return [output_path]

    for scene in scenes:
        start_ms = int(scene["start_time"] * 1000)
        end_ms = int(scene["end_time"] * 1000)
        segment = audio[start_ms:end_ms]

        output_path = os.path.join(output_dir, f"{scene['scene']}.wav")
        segment.export(output_path, format="wav")
        output_files.append(output_path)

    print(f"✅ Extracted {len(output_files)} scene audio files to: {output_dir}")
    return output_files
