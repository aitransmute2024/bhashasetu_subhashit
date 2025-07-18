import os
from moviepy import VideoFileClip


def extract_audio(video_path, audio_path='output/audio.wav'):
    """
    Extracts audio from a video file and saves it as an audio file.

    Args:
        video_path (str): Path to the input video file.
        audio_path (str): Path to save the extracted audio file (e.g., 'output/output.wav').

    Returns:
        str: Path to the saved audio file.
    """
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(audio_path), exist_ok=True)

        # Load video and extract audio
        video = VideoFileClip(video_path)
        if video.audio is None:
            raise ValueError("No audio stream found in the video.")

        video.audio.write_audiofile(audio_path)
        print(f"✅ Audio extracted successfully to: {audio_path}")
        return audio_path

    except Exception as e:
        print(f"❌ Error extracting audio: {e}")
        return None
