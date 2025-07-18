import json
import numpy as np
import os
import torch
import sys

sys.path.append("./FastSpeech2")
from utils.text.symbols import symbols
from text import text_to_sequence
from model import FastSpeech2
from text.symbols import symbols

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load acoustic features
with open("acoustic_features.json", "r") as f:
    data = json.load(f)

# Prepare inputs
text = " ".join([item["word"].strip(",.") for item in data])
sequence = np.array(text_to_sequence(text, ["english_cleaners"]))[None, :]
sequence = torch.from_numpy(sequence).to(device)

# Prepare pitch, energy, duration arrays
def normalize(arr):
    arr = np.array(arr)
    return (arr - arr.mean()) / (arr.std() + 1e-6)

pitch = [item["Pitch (Hz)"] for item in data]
energy = [item["Mean Intensity (dB)"] for item in data]
duration = [round(item["Duration (s)"] * 100) for item in data]  # convert seconds to frame steps

# Normalize pitch and energy
pitch = torch.tensor(normalize(pitch)).float().unsqueeze(0).to(device)
energy = torch.tensor(normalize(energy)).float().unsqueeze(0).to(device)
duration = torch.tensor(duration).unsqueeze(0).to(device)

# Load pretrained FastSpeech2
model = FastSpeech2(len(symbols)).to(device)
model.load_state_dict(torch.load("FastSpeech2/checkpoint.pth.tar")["model"])
model.eval()

# Run model
with torch.no_grad():
    mel = model.inference(
        sequence, duration_control=1.0,
        pitch_control=1.0, energy_control=1.0,
        pitch_target=pitch, energy_target=energy, duration_target=duration
    )[0]

# Save mel for HiFi-GAN
np.save("generated_mel.npy", mel.cpu().numpy(), allow_pickle=False)
print("Mel spectrogram saved. Ready for HiFi-GAN.")

