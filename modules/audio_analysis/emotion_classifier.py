from transformers import pipeline
import soundfile as sf
import librosa

def perform_emotion_analysis(audio_path):
    """
    Performs emotion analysis on an audio file using a pre-trained model.

    Args:
        audio_path (str): Path to the input audio file.

    Returns:
        dict: A dictionary containing the emotion analysis result.
    """
    try:
        # Load audio file
        speech, sample_rate = librosa.load(audio_path, sr=16000)

        # Initialize emotion recognition pipeline
        emotion_recognizer = pipeline("audio-classification", model="superb/wav2vec2-base-superb-er")

        # Perform emotion recognition
        result = emotion_recognizer(speech, top_k=5)

        print(f"\nEmotion Analysis Results for {audio_path}:")
        print(result)
        return result

    except Exception as e:
        print(f"Error during emotion analysis: {e}")
        return None

def perform_sentiment_analysis(audio_path):
    """
    Performs sentiment analysis on an audio file by first transcribing it to text.

    Args:
        audio_path (str): Path to the input audio file.

    Returns:
        dict: A dictionary containing the sentiment analysis result.
    """
    try:
        # Initialize ASR pipeline
        asr_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")

        # Transcribe audio to text
        transcription = asr_pipeline(audio_path)
        text = transcription["text"]

        print(f"\nTranscription for sentiment analysis: {text}")

        # Initialize sentiment analysis pipeline
        sentiment_analyzer = pipeline("sentiment-analysis")

        # Perform sentiment analysis
        result = sentiment_analyzer(text)

        print(f"\nSentiment Analysis Results for {audio_path}:")
        print(result)
        return result

    except Exception as e:
        print(f"Error during sentiment analysis: {e}")
        return None

if __name__ == "__main__":
    audio_file = "output.wav"
    emotion_result = perform_emotion_analysis(audio_file)
    sentiment_result = perform_sentiment_analysis(audio_file)
