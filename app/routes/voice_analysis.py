import os
from preprocess import preprocess_input_file
from modules.audio_analysis.emotion_classifier import perform_emotion_analysis
from modules.audio_analysis.prosodic_feature_extractor import extract_word_level_features


def voice_file_analysis(audio_path):
    """
    Performs voice analysis on a video by preprocessing and analyzing each scene audio.

    Args:
        video_path (str): Path to the input video.
        scene_json (str): Path to scene timestamp JSON file.
        preprocess_function (function): A function that processes the video and returns list of audio paths.

    Returns:
        List[dict]: List of analysis results for each scene.
    """
    emotions = perform_emotion_analysis(audio_path)

    prosodic_features = extract_word_level_features(audio_path)

    return emotions, prosodic_features


