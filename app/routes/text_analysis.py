import os
import asyncio
from preprocess import preprocess_input_file
from modules.text_analysis.asr_transcriber import transcribe_audio
from modules.text_analysis.text_sentiment_analysis import FullTextAnalysis
from modules.text_analysis.translator import translate_text
# from modules.text_analysis.asr_transcriber import transcribe_audio


analyzer = FullTextAnalysis()


def text_file_analysis(audio_path, target_language):
    """
    Performs voice analysis on a video by preprocessing and analyzing each scene audio.

    Args:
        Audip_path (str): Path to the input audio.

    Returns:
        List[dict]: List of analysis results for each scene.
    """
    source_text = transcribe_audio(audio_path)

    # target_text = translate_text(source_text, source_language, target_language)

    target_text = await analyzer.analyze(source_text, target_language)

    sentiment = target_text.get("sentiment")
    emotions = target_text.get("emotions")
    translated_text = target_text.get("translated_text")  # Match the language code used in analyze

    return sentiment, emotions, translated_text, source_text



