import whisper

def transcribe_audio(audio_path):
    model = whisper.load_model("large")  # "large" is a more powerful model compared to "base"
    result = model.transcribe(audio_path)
    return result["text"], result["segments"]

