import asyncio
import nltk
import spacy
import textstat
from langdetect import detect, LangDetectException
from textblob import TextBlob
from rake_nltk import Rake
from keybert import KeyBERT
from transformers import pipeline
from googletrans import Translator, LANGUAGES

# --- NLTK Downloads ---
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)  # Needed for sentence tokenization for potential future use

# --- Global Model/Translator Initializations ---
spacy_model = spacy.load("en_core_web_sm")  # Loads the English small model
translator = Translator()  # Synchronous translator

emotion_model = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=5  # Get top 5 emotions
)

kw_model = KeyBERT()

# Mapping for specific models like MBart (currently unused for translation by googletrans)
LANG_MAP_FOR_SPECIFIC_MODELS = {
    "en": "en_XX", "es": "es_XX", "fr": "fr_XX", "pt": "pt_XX", "it": "it_IT",
    "de": "de_DE", "nl": "nl_XX", "pl": "pl_PL", "ro": "ro_RO", "ru": "ru_RU",
    "uk": "uk_UA", "ar": "ar_AR", "he": "he_IL", "sw": "sw_KE", "hi": "hi_IN",
    "bn": "bn_IN", "gu": "gu_IN", "mr": "mr_IN", "ta": "ta_IN", "te": "te_IN",
    "kn": "kn_IN", "ml": "ml_IN", "pa": "pa_IN", "ur": "ur_PK", "as": "as_IN",
    "zh": "zh_CN", "ja": "ja_XX", "ko": "ko_KR", "id": "id_ID", "vi": "vi_VN",
    "th": "th_TH", "fa": "fa_IR", "tr": "tr_TR", "cs": "cs_CZ",
}


class FullTextAnalysis:
    """
    A class to perform a comprehensive text analysis including language detection,
    sentiment, emotions, keyword extraction, entity recognition, readability,
    and translation. Now includes dependency parsing for contextual understanding.
    """

    def _init_(self):
        self.rake = Rake()
        self.spacy_model = spacy_model
        self.emotion_model = emotion_model
        self.kw_model = kw_model
        self.translator = translator

    def detect_language(self, text: str) -> str:
        """
        Detects the language of the given text using langdetect.
        Returns the ISO 639-1 code (e.g., 'en', 'hi').
        """
        try:
            return detect(text)
        except LangDetectException:
            return "Unknown"
        except Exception:
            return "Unknown"

    def get_sentiment(self, text: str) -> str:
        """
        Calculates the sentiment of the text using TextBlob.
        Returns "Positive", "Negative", or "Neutral".
        """
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        if polarity > 0:
            return "Positive"
        elif polarity < 0:
            return "Negative"
        else:
            return "Neutral"

    def get_emotions(self, text: str) -> list[dict]:
        """
        Detects emotions in the text using a pre-trained Hugging Face model.
        Returns a list of dictionaries with 'label' and 'score'.
        """
        emotions_raw = self.emotion_model(text)
        return emotions_raw[0] if emotions_raw else []

    def extract_rake_keywords(self, text: str) -> list[str]:
        """
        Extracts keywords from the text using RAKE algorithm.
        Returns a list of ranked phrases.
        """
        self.rake.extract_keywords_from_text(text)
        return self.rake.get_ranked_phrases()

    def extract_keybert_keywords(self, text: str, top_n: int = 5) -> list[str]:
        """
        Extracts keywords from the text using KeyBERT.
        Returns a list of top_n keywords.
        """
        return [kw[0] for kw in self.kw_model.extract_keywords(text, top_n=top_n)]

    def get_entities(self, text: str) -> list[tuple[str, str]]:
        """
        Extracts named entities (e.g., persons, organizations) from the text using spaCy.
        Returns a list of (entity_text, entity_label) tuples.
        """
        doc = self.spacy_model(text)
        return [(ent.text, ent.label_) for ent in doc.ents]

    def get_dependency_parse(self, text: str) -> list[dict]:
        """
        Performs dependency parsing using spaCy to identify grammatical relationships
        between words in a sentence.
        Returns a list of dictionaries for each token, showing its text, lemma,
        part-of-speech, dependency relation, head (the word it depends on),
        and its children (words that depend on it).
        """
        doc = self.spacy_model(text)
        parse_results = []
        for token in doc:
            parse_results.append({
                "token": token.text,
                "lemma": token.lemma_,
                "pos": token.pos_,
                "dep": token.dep_,
                "head": token.head.text,
                "children": [child.text for child in token.children]
            })
        return parse_results

    def get_readability(self, text: str) -> str:
        """
        Calculates the Flesch Reading Ease score for the text.
        Returns the score and a readability level (Easy, Medium, Difficult).
        """
        try:
            score = textstat.flesch_reading_ease(text)
            if score > 60:
                return f"{score:.2f} (Easy)"
            elif score > 30:
                return f"{score:.2f} (Medium)"
            else:
                return f"{score:.2f} (Difficult)"
        except Exception:
            return "Not available"

    async def get_translation(self, text: str, target_language: str = 'en', source_language: str = 'auto') -> str:
        """
        Translates the given text to the target language using googletrans.
        This method directly awaits the googletrans.Translator.translate call.
        """
        # The translate method itself returns an awaitable in googletrans 4.x-rc1.
        translation = await self.translator.translate(text, dest=target_language, src=source_language)
        return translation.text

    async def analyze(self, text: str, target_language: str = 'en') -> dict:
        """
        Performs a full analysis of the given text, including language detection,
        sentiment, emotions, keywords (RAKE and KeyBERT), named entities, readability,
        dependency parsing, and translation.
        """
        detected_lang_iso = self.detect_language(text)

        translated_text = await self.get_translation(text, target_language=target_language,
                                                     source_language=detected_lang_iso)

        return {
            "language_detected": detected_lang_iso,
            "sentiment": self.get_sentiment(text),
            "emotions": self.get_emotions(text),
            "keywords_rake": self.extract_rake_keywords(text),
            "keywords_bert": self.extract_keybert_keywords(text),
            "entities": self.get_entities(text),
            "dependency_parse": self.get_dependency_parse(text),  # Added contextual feature
            "readability_score": self.get_readability(text),
            "translated_text": translated_text
        }
