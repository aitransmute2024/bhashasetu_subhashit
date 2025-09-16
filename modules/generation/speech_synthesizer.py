import json
import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
import os
import random

# Load speaker data once
with open("data/speakers.json", "r", encoding="utf-8") as f:
    SPEAKER_DATA = json.load(f)

# Global variables for lazy loading
model = None
tokenizer = None
description_tokenizer = None
device = None


def get_model():
    """
    Lazy-load the ParlerTTS model and tokenizers.
    Allocates GPU memory only when first called.
    """
    global model, tokenizer, description_tokenizer, device
    if model is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            if device.type == "cuda":
                model = ParlerTTSForConditionalGeneration.from_pretrained(
                    "ai4bharat/indic-parler-tts",
                    low_cpu_mem_usage=True
                ).to(device)
            else:
                model = ParlerTTSForConditionalGeneration.from_pretrained(
                    "ai4bharat/indic-parler-tts"
                ).to(device)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("⚠️ CUDA OOM. Falling back to CPU.")
                torch.cuda.empty_cache()
                device = torch.device("cpu")
                model = ParlerTTSForConditionalGeneration.from_pretrained(
                    "ai4bharat/indic-parler-tts"
                ).to(device)
            else:
                raise e

        tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")
        description_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)

    return model, tokenizer, description_tokenizer, device

model, tokenizer, description_tokenizer, device = get_model()

def generate_tts_audio(input_text: str,
                       secs: int,
                       sentiment: str,
                       emotion: str,
                       target_language: str,
                       gender: str,
                       output_file: str = "indic_tts_out.wav") -> str:
    """
    Generates TTS audio for given text, sentiment, emotion, language, and gender.
    Chooses a speaker from speakers.json and saves audio to `output_file`.
    """

    # Get language info
    lang_info = SPEAKER_DATA.get(target_language)
    if not lang_info:
        raise ValueError(f"Language '{target_language}' not found in speaker list.")

    # Filter speakers by gender
    gender_matched_speakers = [sp["name"] for sp in lang_info["available"] if sp["gender"].lower() == gender.lower()]
    if not gender_matched_speakers:
        raise ValueError(f"No speakers found for language '{target_language}' with gender '{gender}'.")

    # Prefer recommended speakers of same gender if available
    recommended_gender_speakers = [sp for sp in lang_info["recommended"] if sp in gender_matched_speakers]
    chosen_speaker = random.choice(recommended_gender_speakers) if recommended_gender_speakers else random.choice(
        gender_matched_speakers)

    # Build dynamic description
    description = (
        f"{chosen_speaker}, a {gender.lower()} speaker, delivers a {sentiment.lower()} "
        f"and {emotion.lower()} speech with natural pitch and pacing. "
        f"The recording is {secs} seconds long, clear and high-quality, in {target_language}."
    )

    # Tokenize
    description_inputs = description_tokenizer(description, return_tensors="pt").to(device)
    prompt_inputs = tokenizer(input_text, return_tensors="pt").to(device)

    # Generate audio
    with torch.no_grad():
        generation = model.generate(
            input_ids=description_inputs["input_ids"],
            attention_mask=description_inputs["attention_mask"],
            prompt_input_ids=prompt_inputs["input_ids"],
            prompt_attention_mask=prompt_inputs["attention_mask"]
        )

    # Convert to numpy and save
    audio_arr = generation.cpu().numpy().squeeze()
    sf.write(output_file, audio_arr, model.config.sampling_rate)

    return os.path.abspath(output_file)


def unload_model():
    """
    Frees GPU memory by deleting model and tokenizers.
    """
    global model, tokenizer, description_tokenizer, device
    del model, tokenizer, description_tokenizer
    model = tokenizer = description_tokenizer = None
    if device and device.type == "cuda":
        torch.cuda.empty_cache()
    device = None


# # Example usage
# if __name__ == "__main__":
#     text = "ਪਾਕਿਸਤਾਨ ਦਾ ਇਹ ਦਾਅਵਾ ਕਿ ਉਨ੍ਹਾਂ ਨੇ ਕਿਸੇ ਵੀ ਧਾਰਮਿਕ ਸਥਾਨ ਨੂੰ ਨਿਸ਼ਾਨਾ ਨਹੀਂ ਬਣਾਇਆ ਜਾਂ ਹਮਲਾ ਨਹੀਂ ਕੀਤਾ..."
#     audio_path = generate_tts_audio(
#         text, secs=15, sentiment="Positive", emotion="Happy", target_language="Punjabi", gender="Male"
#     )
#     print(f"Audio saved at: {audio_path}")
#     unload_model()  # Optional: free GPU memory after TTS


def generate_tts_audio_simple(input_text: str,
                              sentiment: str,
                              target_language: str,
                              gender: str = "female",
                              output_file: str = "indic_tts_out.wav") -> str:
    """
    Generates TTS audio for given text, sentiment, language, and gender.
    Gender defaults to 'female'.
    """

    # Get language info
    lang_info = SPEAKER_DATA.get(target_language)
    if not lang_info:
        raise ValueError(f"Language '{target_language}' not found in speaker list.")

    # Filter speakers by gender
    gender_matched_speakers = [sp["name"] for sp in lang_info["available"] if sp["gender"].lower() == gender.lower()]
    if not gender_matched_speakers:
        raise ValueError(f"No speakers found for language '{target_language}' with gender '{gender}'.")

    # Prefer recommended speakers of same gender if available
    recommended_gender_speakers = [sp for sp in lang_info["recommended"] if sp in gender_matched_speakers]
    chosen_speaker = random.choice(recommended_gender_speakers) if recommended_gender_speakers else random.choice(
        gender_matched_speakers)

    # Build description
    description = (
        f"{chosen_speaker}, a {gender.lower()} speaker, delivers a {sentiment.lower()} "
        f"speech with natural pitch and pacing in {target_language}."
    )

    # Tokenize
    description_inputs = description_tokenizer(description, return_tensors="pt").to(device)
    prompt_inputs = tokenizer(input_text, return_tensors="pt").to(device)

    # Generate audio
    with torch.no_grad():
        generation = model.generate(
            input_ids=description_inputs["input_ids"],
            attention_mask=description_inputs["attention_mask"],
            prompt_input_ids=prompt_inputs["input_ids"],
            prompt_attention_mask=prompt_inputs["attention_mask"]
        )

    # Convert to numpy and save
    audio_arr = generation.cpu().numpy().squeeze()
    sf.write(output_file, audio_arr, model.config.sampling_rate)

    return os.path.abspath(output_file)