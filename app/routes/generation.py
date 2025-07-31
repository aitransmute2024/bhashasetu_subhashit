import os
from modules.generation.aligner import align_sentences
from modules.generation.prosody_mapper import map_target_to_prosodic_features
from modules.generation.speech_synthesizer import generate_tts_audio

def generate_output(src_text, tgt_text, prosodic_features_json, sentiment, emotion, original_duration, target_language, output_path):

    src_json = [entry["word"] for entry in prosodic_features_json if "word" in entry and entry["word"].strip()]

    alignment_json = align_sentences(src_json, tgt_text)
    print(alignment_json)
    target_json = map_target_to_prosodic_features(alignment_json, prosodic_features_json, tgt_text)

    # target_json = [
    #     {
    #         "text": tgt_text,
    #         "pitch_shift": -2,
    #         "gain": 4,
    #         "speed": 1,
    #         "stress": False  # duration = 0.3 > 0.2
    #     }
    # ]

    # final_audio_path = generate_emotional_speech(target_json, emotion, target_language, output_path)
    gender = "Male"

    final_audio_path = generate_tts_audio(
        tgt_text, original_duration, sentiment, emotion, target_language, gender, output_path
    )

    return final_audio_path

