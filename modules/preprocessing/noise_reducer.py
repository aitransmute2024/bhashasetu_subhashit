import os
import librosa
import noisereduce as nr
import soundfile as sf
from scipy.signal import butter, lfilter

def lowpass_filter(data, sr, cutoff_ratio=0.9):
    """
    Applies a low-pass filter to the audio signal.
    Args:
        data (np.array): Audio time series.
        sr (int): Sampling rate.
        cutoff_ratio (float): Ratio (0-1) to determine cutoff frequency relative to Nyquist.
    Returns:
        np.array: Filtered audio signal.
    """
    nyquist = sr / 2
    cutoff = cutoff_ratio * nyquist
    b, a = butter(N=6, Wn=cutoff / nyquist, btype='low', analog=False)
    return lfilter(b, a, data)

def clean_audio(input_path, output_path='output/cleaned_audio.wav'):
    """
    Performs noise reduction and low-pass filtering on input audio.

    Args:
        input_path (str): Path to the raw input audio file.
        output_path (str): Path to save the cleaned audio file.

    Returns:
        str: Path to the saved cleaned audio file.
    """
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Load audio
        y, sr = librosa.load(input_path, sr=16000, mono=True)

        # Step 1: Noise reduction
        y_denoised = nr.reduce_noise(y=y, sr=sr, prop_decrease=1.0)

        # Step 2: Apply low-pass filter
        y_smoothed = lowpass_filter(y_denoised, sr)

        # Save the cleaned audio
        sf.write(output_path, y_smoothed, sr)

        print(f"✅ Cleaned audio saved at: {output_path}")
        return output_path

    except Exception as e:
        print(f"❌ Error cleaning audio: {e}")
        return None
