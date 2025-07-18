import librosa
import numpy as np
from pydub import AudioSegment

def get_pitch_and_loudness(audio_path):
    """
    Extracts pitch (F0) and loudness (RMS energy) from an audio file.

    Args:
        audio_path (str): Path to the input audio file.

    Returns:
        dict: A dictionary containing pitch and loudness information.
    """
    try:
        y, sr = librosa.load(audio_path)

        # Pitch (F0) estimation
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))

        # Loudness (RMS energy)
        rms = librosa.feature.rms(y=y)[0]

        return {
            "pitch_f0": f0.tolist(),
            "pitch_voiced_flag": voiced_flag.tolist(),
            "pitch_voiced_probs": voiced_probs.tolist(),
            "loudness_rms": rms.tolist()
        }
    except Exception as e:
        print(f"Error extracting pitch and loudness: {e}")
        return None

def get_speaking_rate(audio_path, transcription_text):
    """
    Estimates speaking rate (words per minute) from an audio file and its transcription.

    Args:
        audio_path (str): Path to the input audio file.
        transcription_text (str): Transcribed text of the audio.

    Returns:
        float: Speaking rate in words per minute.
    """
    try:
        audio = AudioSegment.from_file(audio_path)
        duration_minutes = len(audio) / (1000 * 60)  # Duration in minutes
        word_count = len(transcription_text.split())
        speaking_rate = word_count / duration_minutes if duration_minutes > 0 else 0
        return speaking_rate
    except Exception as e:
        print(f"Error estimating speaking rate: {e}")
        return None

def get_intonation(pitch_f0):
    """
    Placeholder for intonation analysis. This would involve analyzing pitch contours.

    Args:
        pitch_f0 (list): List of fundamental frequencies.

    Returns:
        str: A placeholder string for intonation analysis.
    """
    # This is a complex task and would require more sophisticated analysis of pitch contours.
    # For now, we'll return a placeholder.
    return "Intonation analysis requires further implementation based on pitch contours."

def get_voice_quality(audio_path):
    """
    Placeholder for voice quality analysis. This would involve features like jitter, shimmer, HNR.

    Args:
        audio_path (str): Path to the input audio file.

    Returns:
        str: A placeholder string for voice quality analysis.
    """
    # Voice quality features like jitter, shimmer, HNR are typically extracted using specialized tools
    # or libraries that implement these algorithms. librosa does not directly provide them.
    return "Voice quality analysis requires specialized libraries or custom implementation."

if __name__ == "__main__":
    audio_file = "output.wav"
    pitch_loudness = get_pitch_and_loudness(audio_file)
    print(f"Pitch and Loudness: {pitch_loudness}")

    transcription = "This is an example sentence for speaking rate calculation."
    speaking_rate = get_speaking_rate(audio_file, transcription)
    print(f"Speaking Rate: {speaking_rate} words per minute")

    intonation = get_intonation(pitch_loudness["pitch_f0"] if pitch_loudness else [])
    print(f"Intonation: {intonation}")

    voice_quality = get_voice_quality(audio_file)
    print(f"Voice Quality: {voice_quality}")
    
