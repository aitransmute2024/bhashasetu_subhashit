import whisper

def transcribe_audio(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result["text"], result["segments"]

# Example usage:
if __name__ == "__main__":
    text, segments = transcribe_audio("scenes_audio/scene2.wav")
    print("Transcribed Text:", text)