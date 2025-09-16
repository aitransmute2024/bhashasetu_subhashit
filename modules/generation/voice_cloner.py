# # import torchaudio as ta
# # from chatterbox.tts import ChatterboxTTS
# #
# # model = ChatterboxTTS.from_pretrained(device="cpu")
# #
# # text = "Ezreal and Jinx teamed up with Ahri, Yasuo, and Teemo to take down the enemy's Nexus in an epic late-game pentakill."
# # # wav = model.generate(text)
# # # ta.save("test-1.wav", wav, model.sr)
# #
# # # If you want to synthesize with a different voice, specify the audio prompt
# # AUDIO_PROMPT_PATH = r'C:\Users\admin\OneDrive - Aidwise Private Ltd\BhashaSetu_VAM\samples\source.wav'
# # wav = model.generate(text, audio_prompt_path=AUDIO_PROMPT_PATH)
# # ta.save("test-2.wav", wav, model.sr)
#
#
# from transformers import AutoModel
# import numpy as np
# import soundfile as sf
#
# # Load INF5 from Hugging Face
# repo_id = "ai4bharat/IndicF5"
# model = AutoModel.from_pretrained(repo_id, trust_remote_code=True)
#
# # Generate speech
# audio = model(
#     "नमस्ते! संगीत की तरह जीवन भी खूबसूरत होता है, बस इसे सही ताल में जीना आना चाहिए.",
#     ref_audio_path="C:/Users/admin/OneDrive - Aidwise Private Ltd/BhashaSetu_VAM/samples/source-trimmed.wav",
#     ref_text="What I want to, underline is that the official and blatantly farcical denial of these attacks that Pakistan carried out by the Pakistani state machinery."
# )
#
# # Normalize and save output
# if audio.dtype == np.int16:
#     audio = audio.astype(np.float32) / 32768.0
# sf.write("namaste.wav", np.array(audio, dtype=np.float32), samplerate=24000)