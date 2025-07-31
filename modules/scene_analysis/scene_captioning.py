# blip_captioner.py
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import os
import glob

def caption_image(image_path, processor, model):
    image = Image.open(image_path).convert('RGB')
    inputs = processor(image, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption

if __name__ == "__main__":
    image_folder = "scene_frames"
    image_paths = sorted(glob.glob(os.path.join(image_folder, "scene_*.jpg")))

    if not image_paths:
        print("[ERROR] No scene images found. Run 'scene_analyzer.py' first.")
        exit()

    print("[INFO] Loading BLIP model...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    print("\n[CAPTIONS]")
    for image_path in image_paths:
        caption = caption_image(image_path, processor, model)
        print(f"{os.path.basename(image_path)} → {caption}")
