import os
from modules.generation.aligner import align_sentences
from modules.generation.prosody_mapper import map_target_to_prosodic_features
from modules.generation.speech_synthesizer import generate_emotional_speech

def generate_output(src_text, tgt_text, prosodic_features_json, emotion, target_language, output_path):

    alignment_json = align_sentences(src_text, tgt_text)

    target_json = map_target_to_prosodic_features(alignment_json,prosodic_features_json,tgt_text)

    final_audio_path = generate_emotional_speech(target_json, emotion, target_language, output_path)

    return final_audio_path

