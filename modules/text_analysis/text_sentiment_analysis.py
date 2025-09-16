import asyncio
import concurrent.futures
import torch
import spacy
import textstat
from langdetect import detect
from textblob import TextBlob
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from typing import Dict

# Load SpaCy for lightweight tasks (NER removed, only language utilities if needed)
spacy_model = spacy.load("en_core_web_sm")

# Load GPU if available
DEVICE = 0 if torch.cuda.is_available() else -1

# Load Emotion classification model (GPU if available)
emotion_model = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=5,
    device=DEVICE
)


class UnifiedTextAnalysis:
    def __init__(self):
        self.spacy_model = spacy_model
        self.emotion_model = emotion_model

        # Load NLLB-200 (distilled 600M for balance, can switch to 1.3B for higher BLEU)
        model_name = "facebook/nllb-200-distilled-600M"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

    # --- Core NLP Methods ---

    def detect_language(self, text: str) -> str:
        try:
            return detect(text)
        except Exception:
            return "unknown"

    def get_sentiment(self, text: str) -> str:
        polarity = TextBlob(text).sentiment.polarity
        return "Positive" if polarity > 0 else "Negative" if polarity < 0 else "Neutral"

    def get_emotions(self, text: str):
        try:
            return self.emotion_model(text)[0]
        except Exception:
            return []

    def get_readability(self, text: str) -> str:
        try:
            score = textstat.flesch_reading_ease(text)
            if score > 60:
                level = "Easy"
            elif score > 30:
                level = "Medium"
            else:
                level = "Difficult"
            return f"{score:.2f} ({level})"
        except Exception:
            return "Not available"

    # --- Translation with NLLB ---

    def get_nllb_lang_code(self, lang: str) -> str:
        mapping = {
            'en': 'eng_Latn', 'hi': 'hin_Deva', 'bn': 'ben_Beng', 'ta': 'tam_Taml',
            'te': 'tel_Telu', 'ml': 'mal_Mlym', 'kn': 'kan_Knda', 'gu': 'guj_Gujr',
            'mr': 'mar_Deva', 'pa': 'pan_Guru', 'ur': 'urd_Arab', 'as': 'asm_Beng',
            'or': 'ory_Orya', 'ne': 'npi_Deva', 'fr': 'fra_Latn', 'de': 'deu_Latn',
            'es': 'spa_Latn'
        }
        return mapping.get(lang, 'eng_Latn')

    async def translate_text(self, text: str, source_lang='auto', target_lang='en') -> str:
        if source_lang == 'auto':
            source_lang = self.detect_language(text)

        src = self.get_nllb_lang_code(source_lang)
        tgt = self.get_nllb_lang_code(target_lang)

        def _translate():
            translator = pipeline(
                "translation",
                model=self.model,
                tokenizer=self.tokenizer,
                src_lang=src,
                tgt_lang=tgt,
                device=DEVICE
            )
            result = translator(text, max_length=512)
            return result[0]['translation_text']

        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return await loop.run_in_executor(pool, _translate)

    # --- Full Analysis ---

    async def analyze(self, text: str, target_language='en') -> Dict:
        if isinstance(text, tuple):
            text = text[0]

        detected_lang = self.detect_language(text)
        translated_text = await self.translate_text(
            text,
            source_lang=detected_lang,
            target_lang=target_language
        )

        return {
            "language_detected": detected_lang,
            "sentiment": self.get_sentiment(text),
            "emotions": self.get_emotions(text),
            "readability_score": self.get_readability(text),
            "translated_text": translated_text
        }


# # Example Usage
# if __name__ == "__main__":
#     async def run():
#         analyzer = UnifiedTextAnalysis()
#         result = await analyzer.analyze(
#             "मैं इस पूरी स्थिति से बेहद परेशान हूँ।",
#             target_language="en"
#         )
#         print(result)

#     asyncio.run(run())
