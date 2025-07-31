# import torchaudio
# from speechbrain.inference import EncoderClassifier
# import numpy as np
#
# # Load speaker embedding model
# classifier = EncoderClassifier.from_hparams(
#     source="speechbrain/spkrec-ecapa-voxceleb",
#     savedir="pretrained_models/spkrec-ecapa"
# )
# path = r'C:\Users\admin\OneDrive - Aidwise Private Ltd\BhashaSetu_VAM\samples\full_cleaned_audio.wav'
# signal, fs = torchaudio.load(path)
# embeddings = classifier.encode_batch(signal).detach().numpy()
#
# # Placeholder: Gender classification based on pitch or trained model
# mean_pitch = np.mean(signal.numpy())
# if mean_pitch > 0.05:
#     print("Likely Female")
# else:
#     print("Likely Male")
