import os
from pydub import AudioSegment
from .text_analysis import text_file_analysis
from .voice_analysis import voice_file_analysis
from .generation import generate_output
from modules.preprocessing.video_segmenter import extract_scenes
from modules.preprocessing.noise_reducer import clean_audio
from modules.preprocessing.audio_splitter import split_audio_by_scenes
from modules.preprocessing.audio_extractor import extract_audio
from modules.audio_analysis.diarization import diarize_and_extract_speakers
from difflib import get_close_matches

# Language name to short code mapping
LANGUAGE_MAP = {
    'hindi': 'hi',
    'bengali': 'bn',
    'telugu': 'te',
    'marathi': 'mr',
    'tamil': 'ta',
    'urdu': 'ur',
    'gujarati': 'gu',
    'kannada': 'kn',
    'malayalam': 'ml',
    'punjabi': 'pa',
    'odia': 'or',
    'oriya': 'or',
    'assamese': 'as',
}


def get_language_code(input_language):
    input_language = input_language.strip().lower()
    if input_language in LANGUAGE_MAP:
        return LANGUAGE_MAP[input_language]

    # Fuzzy matching for minor typos
    match = get_close_matches(input_language, LANGUAGE_MAP.keys(), n=1, cutoff=0.6)
    if match:
        return LANGUAGE_MAP[match[0]]

    raise ValueError(f"Language '{input_language}' is not supported or misspelled.")

def complete_pipeline(file_path, target_language):

    target_language = get_language_code(target_language)

    audio_path = extract_audio(file_path)

    cleaned_audio_path = clean_audio(audio_path)

    speaker_data_json = diarize_and_extract_speakers(cleaned_audio_path)

    base_audio = AudioSegment.from_wav(cleaned_audio_path)

    output_segments = []

    # Iterate through each speaker in order
    for speaker_id, data in speaker_data_json.items():
        for seg in data['segments']:
            start_ms = int(seg['start'] * 1000)
            end_ms = int(seg['end'] * 1000)

            segment_audio = base_audio[start_ms:end_ms]
            segment_path = f'temp_segments/{speaker_id}_{start_ms}_{end_ms}.wav'
            os.makedirs('temp_segments', exist_ok=True)
            segment_audio.export(segment_path, format='wav')

            # Process each segment separately
            emotions, prosodic_features = voice_file_analysis(segment_path)
            sentiment, emotions, translated_text, source_text = text_file_analysis(segment_path, target_language)
            emotions = "happy"
            # Generate processed audio for this segment
            output_path = f'temp_segments/processed_{speaker_id}_{start_ms}_{end_ms}.wav'
            generate_output(source_text, translated_text, prosodic_features, emotions, target_language, output_path)

            output_segments.append((start_ms, output_path))

    # Process pause segments
    for pause in speaker_data_json.get("pause_segments", []):
        start_ms = int(pause['start'] * 1000)
        end_ms = int(pause['end'] * 1000)
        duration = end_ms - start_ms
        pause_audio = AudioSegment.silent(duration=duration)
        pause_path = f'temp_segments/pause_{start_ms}_{end_ms}.wav'
        pause_audio.export(pause_path, format='wav')
        output_segments.append((start_ms, pause_path))

    # Sort all segments by start time
    output_segments.sort(key=lambda x: x[0])

    # Combine all processed audios
    final_audio = AudioSegment.empty()
    for _, path in output_segments:
        final_audio += AudioSegment.from_wav(path)

    # Export final audio
    os.makedirs('final_output', exist_ok=True)
    final_audio.export('final_output/final_audio_file.wav', format='wav')
    final_audio_path = 'final_audio_path'
    return final_audio_path




