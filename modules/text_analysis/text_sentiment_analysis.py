# ---- Imports ----
import nltk
import spacy
import textstat
import asyncio
import nest_asyncio
from langdetect import detect, LangDetectException
from textblob import TextBlob
from rake_nltk import Rake
from keybert import KeyBERT
from transformers import pipeline
from googletrans import Translator, LANGUAGES
import concurrent.futures

# ---- Setup ----
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nest_asyncio.apply()

# ---- Models ----
spacy_model = spacy.load("en_core_web_sm")
translator = Translator()
emotion_model = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=5)
kw_model = KeyBERT()

# ---- Class ----
class FullTextAnalysis:
    def __init__(self):
        self.rake = Rake()
        self.spacy_model = spacy_model
        self.emotion_model = emotion_model
        self.kw_model = kw_model
        self.translator = translator

    def detect_language(self, text: str) -> str:
        try:
            return detect(text)
        except (LangDetectException, Exception):
            return "unknown"

    def get_sentiment(self, text: str) -> str:
        blob = TextBlob(str(text))
        polarity = blob.sentiment.polarity
        return "Positive" if polarity > 0 else "Negative" if polarity < 0 else "Neutral"

    def get_emotions(self, text: str) -> list[dict]:
        try:
            result = self.emotion_model(text)
            return result[0] if result else []
        except Exception:
            return []

    def extract_rake_keywords(self, text: str) -> list[str]:
        self.rake.extract_keywords_from_text(text)
        return self.rake.get_ranked_phrases()

    def extract_keybert_keywords(self, text: str, top_n: int = 5) -> list[str]:
        try:
            return [kw[0] for kw in self.kw_model.extract_keywords(text, top_n=top_n)]
        except Exception:
            return []

    def get_entities(self, text: str) -> list[tuple[str, str]]:
        doc = self.spacy_model(text)
        return [(ent.text, ent.label_) for ent in doc.ents]

    def get_dependency_parse(self, text: str) -> list[dict]:
        doc = self.spacy_model(text)
        return [{
            "token": token.text,
            "lemma": token.lemma_,
            "pos": token.pos_,
            "dep": token.dep_,
            "head": token.head.text,
            "children": [child.text for child in token.children]
        } for token in doc]

    def get_readability(self, text: str) -> str:
        try:
            score = textstat.flesch_reading_ease(text)
            level = "Easy" if score > 60 else "Medium" if score > 30 else "Difficult"
            return f"{score:.2f} ({level})"
        except Exception:
            return "Not available"

    async def get_translation(self, text: str, target_language: str = 'en', source_language: str = 'auto') -> str:
        try:
            if source_language == 'auto' or source_language not in LANGUAGES:
                source_language = None  # auto-detect
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                lambda: self.translator.translate(text, src=source_language, dest=target_language).text
            )
        except Exception as e:
            print(f"[Translation Error] {e}")
            return text

    async def analyze(self, text: str, target_language: str = 'en') -> dict:
        if isinstance(text, tuple):
            text = text[0]

        detected_lang = self.detect_language(text)
        translated_text = await self.get_translation(text, target_language=target_language, source_language=detected_lang)

        return {
            "language_detected": detected_lang,
            "sentiment": self.get_sentiment(text),
            "emotions": self.get_emotions(text),
            "keywords_rake": self.extract_rake_keywords(text),
            "keywords_bert": self.extract_keybert_keywords(text),
            "entities": self.get_entities(text),
            "dependency_parse": self.get_dependency_parse(text),
            "readability_score": self.get_readability(text),
            "translated_text": translated_text
        }

# Example usage
# async def main():
#     analyzer = FullTextAnalysis()
#     result = await analyzer.analyze("Bonjour, comment allez-vous aujourd'hui ?", target_language="ta")
#
#     # Display result
#     for key, value in result.items():
#         print(f"{key}:\n{value}\n{'-'*50}")
#
# if __name__ == "__main__":
#     asyncio.run(main())


