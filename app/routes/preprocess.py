import os
import mimetypes
from moviepy import VideoFileClip
from pydub import AudioSegment
from modules.preprocessing.video_segmenter import extract_scenes
from modules.preprocessing.noise_reducer import clean_audio
from modules.preprocessing.audio_splitter import split_audio_by_scenes
from modules.preprocessing.audio_extractor import extract_audio

def preprocess_input_file(file_path: str, output_dir="output/processed"):
    """
    Preprocess input file (video/audio/text).
    Converts video to .mp4, audio to .wav, and reads text.

    Args:
        file_path (str): Path to input file.
        output_dir (str): Directory to save processed files.

    Returns:
        dict: Dictionary with keys 'type', 'path', and optionally 'text'.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    os.makedirs(output_dir, exist_ok=True)
    mime_type, _ = mimetypes.guess_type(file_path)

    if mime_type is None:
        raise ValueError("Unable to determine file type.")

    if mime_type.startswith("video"):
        ext = os.path.splitext(file_path)[-1].lower()
        if ext != ".mp4":
            output_path = os.path.join(output_dir, os.path.splitext(os.path.basename(file_path))[0] + ".mp4")
            print(f"Converting video to mp4: {output_path}")
            clip = VideoFileClip(file_path)
            clip.write_videofile(output_path, codec="libx264", audio_codec="aac", verbose=False, logger=None)
            file_path = output_path

        scenes_json = extract_scenes(file_path)

        audio_path = extract_audio(file_path)

        cleaned_audio_path = clean_audio(audio_path)

        scenes_path = split_audio_by_scenes(cleaned_audio_path, scenes_json)

        return scenes_path

    elif mime_type.startswith("audio"):
        ext = os.path.splitext(file_path)[-1].lower()
        if ext != ".wav":
            audio = AudioSegment.from_file(file_path)
            output_path = os.path.join(output_dir, os.path.splitext(os.path.basename(file_path))[0] + ".wav")
            print(f"Converting audio to wav: {output_path}")
            audio.export(output_path, format="wav")
            return {"type": "audio", "path": output_path}
        else:
            return {"type": "audio", "path": file_path}

    elif mime_type.startswith("text") or file_path.endswith(('.srt', '.vtt', '.txt')):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return {"type": "text", "path": file_path, "text": content}

    else:
        raise ValueError(f"Unsupported file type: {mime_type}")
