import os
import asyncio
from modules.text_analysis.asr_transcriber import transcribe_audio
from modules.text_analysis.text_sentiment_analysis import UnifiedTextAnalysis
# from modules.text_analysis.asr_transcriber import transcribe_audio
import asyncio
from modules.text_analysis.phrase_swapping import process_text

def safe_async_call(coro):
    try:
        return asyncio.run(coro)
    except RuntimeError as e:
        if "already running" in str(e):
            import nest_asyncio
            nest_asyncio.apply()
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(coro)
        else:
            raise

# Usage
analyzer = UnifiedTextAnalysis(translation_backend="nllb")

def text_file_analysis(audio_path, target_language):
    """
    Performs voice analysis on a video by preprocessing and analyzing each scene audio.

    Args:
        Audip_path (str): Path to the input audio.

    Returns:
        List[dict]: List of analysis results for each scene.
    """
    source_text, source_segments = transcribe_audio(audio_path)

    phrase_swap_text = process_text(source_text)

    target_text = safe_async_call(analyzer.analyze(phrase_swap_text, target_language))

    sentiment = target_text.get("sentiment")
    emotions = target_text.get("emotions")
    major_emotion = max(emotions, key=lambda x: x['score'])['label']
    translated_text = target_text.get("translated_text")  # Match the language code used in analyze

    return sentiment, major_emotion, translated_text, source_text



