import asyncio
import concurrent.futures
import spacy
spacy_model = spacy.load("en_core_web_sm")
import textstat
from langdetect import detect
from textblob import TextBlob
from rake_nltk import Rake
from keybert import KeyBERT
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import argostranslate.package
import argostranslate.translate
from googletrans import Translator, LANGUAGES
from modules.text_analysis.translator import detect_and_translate
# NLTK Setup
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

# Load NLP models
emotion_model = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=5)
kw_model = KeyBERT()

class UnifiedTextAnalysis:
    def __init__(self, translation_backend="nllb"):
        self.rake = Rake()
        self.spacy_model = spacy_model
        self.emotion_model = emotion_model
        self.kw_model = kw_model
        self.translation_backend = translation_backend

        # Translation setup
        self.translator_google = Translator()

        if translation_backend == "nllb":
            self.tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
            self.model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
        elif translation_backend == "argos":
            argostranslate.package.update_package_index()
            self.available_packages = argostranslate.package.get_available_packages()

    # --- Core NLP Methods ---

    def detect_language(self, text):
        try:
            return detect(text)
        except Exception:
            return "unknown"

    def get_sentiment(self, text):
        polarity = TextBlob(text).sentiment.polarity
        return "Positive" if polarity > 0 else "Negative" if polarity < 0 else "Neutral"

    def get_emotions(self, text):
        try:
            return self.emotion_model(text)[0]
        except Exception:
            return []

    def extract_rake_keywords(self, text):
        self.rake.extract_keywords_from_text(text)
        return self.rake.get_ranked_phrases()

    def extract_keybert_keywords(self, text, top_n=5):
        try:
            return [kw[0] for kw in self.kw_model.extract_keywords(text, top_n=top_n)]
        except Exception:
            return []

    def get_entities(self, text):
        if not self.spacy_model:
            return []
        doc = self.spacy_model(text)
        return [(ent.text, ent.label_) for ent in doc.ents]

    def get_dependency_parse(self, text):
        if not self.spacy_model:
            return []
        doc = self.spacy_model(text)
        return [{
            "token": token.text,
            "lemma": token.lemma_,
            "pos": token.pos_,
            "dep": token.dep_,
            "head": token.head.text,
            "children": [child.text for child in token.children]
        } for token in doc]

    def get_readability(self, text):
        if not textstat:
            return "Not available"
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

    # --- Translation Backends ---

    def get_nllb_lang_code(self, lang):
        mapping = {
            'en': 'eng_Latn', 'hi': 'hin_Deva', 'bn': 'ben_Beng', 'ta': 'tam_Taml',
            'te': 'tel_Telu', 'ml': 'mal_Mlym', 'kn': 'kan_Knda', 'gu': 'guj_Gujr',
            'mr': 'mar_Deva', 'pa': 'pan_Guru', 'ur': 'urd_Arab', 'as': 'asm_Beng',
            'or': 'ory_Orya', 'ne': 'nep_Deva', 'fr': 'fra_Latn', 'de': 'deu_Latn',
            'es': 'spa_Latn'
        }
        return mapping.get(lang, 'eng_Latn')

    async def translate_text(self, text, source_lang='auto', target_lang='en'):
        if source_lang == 'auto':
            source_lang = self.detect_language(text)

        loop = asyncio.get_event_loop()

        if self.translation_backend == "google":
            with concurrent.futures.ThreadPoolExecutor() as pool:
                translation = await loop.run_in_executor(pool, lambda: self.translator_google.translate(text, dest=target_lang, src=source_lang))
                return translation.text

        elif self.translation_backend == "nllb":
            src = self.get_nllb_lang_code(source_lang)
            tgt = self.get_nllb_lang_code(target_lang)

            def _translate():
                translator = pipeline("translation", model=self.model, tokenizer=self.tokenizer, src_lang=src, tgt_lang=tgt)
                result = translator(text, max_length=512)
                return result[0]['translation_text']

            with concurrent.futures.ThreadPoolExecutor() as pool:
                return await loop.run_in_executor(pool, _translate)

        elif self.translation_backend == "argos":
            def _translate():
                package = next((pkg for pkg in self.available_packages if pkg.from_code == source_lang and pkg.to_code == target_lang), None)
                if package and not package.is_installed():
                    argostranslate.package.install_from_path(package.download())
                return argostranslate.translate.translate(text, source_lang, target_lang)

            with concurrent.futures.ThreadPoolExecutor() as pool:
                return await loop.run_in_executor(pool, _translate)

        return text  # fallback

    # --- Full Analysis ---

    async def analyze(self, text, target_language='en'):
        if isinstance(text, tuple):
            text = text[0]

        detected_lang = self.detect_language(text)
        translated_text = await self.translate_text(text, source_lang=detected_lang, target_lang=target_language)

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

# # Example Usage
# if __name__ == "__main__":
#     async def run():
#         analyzer = UnifiedTextAnalysis(translation_backend="nllb")
#         result = await analyzer.analyze("I’m really frustrated with how things are being handled — this is completely unacceptable!.", target_language="hi")
#         print(result)
#
#     asyncio.run(run())
