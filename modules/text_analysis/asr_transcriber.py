import torch
import torchaudio
from transformers import pipeline
torch.cuda.empty_cache()
# Load ASR pipeline (Whisper Large V3, or replace with IndicWhisper if needed)
model_path = "openai/whisper-medium"
asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model=model_path,
    device=0 if torch.cuda.is_available() else -1
)
print("Using device:", asr_pipeline.device)

# Ensure no forced language
asr_pipeline.model.config.forced_decoder_ids = None


def transcribe_audio(audio_path: str, sr: int = 16000):
    """
    Transcribe an audio file into text and return text + segments.
    
    Args:
        audio_path (str): Path to the audio file.
        sr (int): Target sampling rate for Whisper (default=16k).
    
    Returns:
        tuple: (full_text, segments)
    """
    # Load audio
    wav, input_sr = torchaudio.load(audio_path)

    # Resample if needed
    if input_sr != sr:
        resampler = torchaudio.transforms.Resample(input_sr, sr)
        wav = resampler(wav)

    # Convert to mono if stereo
    if wav.shape[0] > 1:
        wav = torch.mean(wav, dim=0, keepdim=True)

    # Convert tensor -> numpy
    waveform = wav.numpy().flatten()

    # Run ASR
    result = asr_pipeline(
        waveform,
        chunk_length_s=30,
        return_timestamps="word"
    )

    # Extract transcription and segments
    full_text = result.get("text", "")
    segments = result.get("chunks", [])

    return full_text, segments


# # Example usage
# if __name__ == "__main__":
#     audio_file = r'C:\Users\Sidhant Raj\Desktop\Sidhant\bhashasetu_subhashit\samples\source.wav'
#     text, segments = transcribe_audio(audio_file)

#     print("Full Transcription:\n", text)
#     print("\nWord/Chunk Segments:")
#     for seg in segments[:10]:  # print only first 10 for readability
#         print(seg)
