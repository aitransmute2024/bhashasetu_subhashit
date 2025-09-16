import os
import json
import time
import re
import requests
import asyncio
import concurrent.futures
import torch
from langdetect import detect
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List, Dict
import nest_asyncio

nest_asyncio.apply()

# --- Configuration ---
GEMINI_API_KEY = "AIzaSyCx7yd6sXj2Mc8VsH6U3JtomB36jBCgMMc"
KNOWLEDGE_BASE_DB_FILE = r'data/knowledge_base.json'


# --- Main Hybrid Translation Class ---

class HybridTranslationSystem:
    def __init__(self):
        """Initializes the system, loading all necessary models."""
        print("Initializing Hybrid Translation System...")
        # Determine device (GPU or CPU) for PyTorch models
        self.device_id = 0 if torch.cuda.is_available() else -1
        self.torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.torch_device}")

        # --- Load Local NLLB Model ---
        print("Loading Facebook NLLB-200 model... (This may take a moment)")
        nllb_model_name = "facebook/nllb-200-distilled-600M"
        self.nllb_tokenizer = AutoTokenizer.from_pretrained(nllb_model_name)
        self.nllb_model = AutoModelForSeq2SeqLM.from_pretrained(nllb_model_name).to(self.torch_device)
        print("✅ NLLB model loaded successfully.")

    # --- Knowledge Base Methods ---
    def _load_figurative_speech_db(self) -> List[Dict]:
        """Loads the figurative speech database from the JSON file."""
        if not os.path.exists(KNOWLEDGE_BASE_DB_FILE):
            return []
        with open(KNOWLEDGE_BASE_DB_FILE, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                print(f"⚠️ Warning: Could not decode {KNOWLEDGE_BASE_DB_FILE}. Starting with an empty DB.")
                return []

    def _save_figurative_speech_db(self, db: List[Dict]):
        """Saves the figurative speech database to the JSON file."""
        with open(KNOWLEDGE_BASE_DB_FILE, 'w', encoding='utf-8') as f:
            json.dump(db, f, indent=4, ensure_ascii=False)

    # --- API Calling Methods ---
    def _call_gemini_api_sync(self, prompt: str) -> str:

        """Synchronous, robust Gemini API call using the requests library."""
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
        headers = {"Content-Type": "application/json"}
        try:
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.1}
            }
            response = requests.post(api_url, headers=headers, data=json.dumps(payload), timeout=60)
            response.raise_for_status()
            result = response.json()
            return result['candidates'][0]['content']['parts'][0]['text'].strip()
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Gemini request failed: {e}")
            return "Error: Gemini API call failed."
        except (KeyError, IndexError) as e:
            print(f"⚠️ Unexpected Gemini response format: {e}")
            return "Error: Could not parse Gemini response."

    async def _call_gemini_api_async(self, prompt: str) -> str:
        """Asynchronous wrapper to run the synchronous Gemini API call in a separate thread."""
        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return await loop.run_in_executor(pool, self._call_gemini_api_sync, prompt)

    # --- Core Logic Methods ---
    async def _detect_figurative_speech(self, text: str) -> int:
        """Detects if a sentence contains figurative speech, checking the local DB first."""
        db = self._load_figurative_speech_db()
        for entry in db:
            if re.search(r'\b' + re.escape(entry["figurative_speech"].lower()) + r'\b', text.lower()):
                print(f"✅ Figurative speech found in DB: '{entry['figurative_speech']}'")
                return 1

        prompt = f"Does the sentence contain figurative speech (like an idiom or metaphor)? Respond with only '1' for yes or '0' for no. Sentence: \"{text}\""
        print("🤖 Checking for figurative speech via Gemini API...")
        response = await self._call_gemini_api_async(prompt)
        return 1 if response.strip() == '1' else 0

    async def _handle_figurative_translation(self, statement: str, target_language: str) -> str:
        """Translates figurative speech, learning new phrases and saving them to the DB."""
        db = self._load_figurative_speech_db()
        figurative_phrase, literal_meaning = None, None

        # Step 1: Check if the phrase is already in our local database
        for entry in db:
            if re.search(r'\b' + re.escape(entry["figurative_speech"].lower()) + r'\b', statement.lower()):
                figurative_phrase = entry["figurative_speech"]
                literal_meaning = entry["literal_meaning"]
                print(f"📚 Meaning for '{figurative_phrase}' found in local DB.")
                break

        # Step 2: If not in DB, use Gemini to learn it
        if not figurative_phrase:
            print("🤖 Phrase not in DB. Asking Gemini to identify and explain...")
            identify_prompt = f"""Identify the figurative speech phrase in the following sentence and provide its literal meaning.
Respond ONLY with a valid JSON object in this format:
{{
  "figurative_speech": "the specific phrase found",
  "literal_meaning": "the literal meaning of that phrase"
}}

Sentence: "{statement}"
"""
            response_str = await self._call_gemini_api_async(identify_prompt)

            try:
                # Clean Gemini's response (removes markdown ```json fences)
                cleaned_json_str = re.sub(r"```(?:json)?", "", response_str).strip()
                data = json.loads(cleaned_json_str)
                figurative_phrase = data.get("figurative_speech")
                literal_meaning = data.get("literal_meaning")

                if figurative_phrase and literal_meaning:
                    print(f"🧠 Gemini identified: '{figurative_phrase}' -> '{literal_meaning}'")
                    new_entry = {"figurative_speech": figurative_phrase, "literal_meaning": literal_meaning}
                    db.append(new_entry)
                    self._save_figurative_speech_db(db)
                    print(f"💾 New phrase saved to '{KNOWLEDGE_BASE_DB_FILE}'.")
                else:
                    print("⚠️ Gemini could not identify a phrase. Proceeding with direct translation.")
            except (json.JSONDecodeError, AttributeError):
                print("⚠️ Failed to parse identification from Gemini. Proceeding with direct translation.")

        # Step 3: Perform the translation using the context we found or learned
        if figurative_phrase and literal_meaning:
            print("🌐 Translating with context using Gemini...")
            translation_prompt = f"""Translate the following English sentence to {target_language}.
Pay close attention to the figurative phrase "{figurative_phrase}", which has a literal meaning of "{literal_meaning}".
Use this context to create a translation that sounds natural and conveys the correct meaning in {target_language}. Do not explain your work, just provide the final translation.

English Sentence: "{statement}"
"""
        else:
            # Fallback if no figurative phrase was identified
            print("🌐 No context available. Performing standard translation with Gemini.")
            translation_prompt = f'Translate the following English sentence to {target_language}: "{statement}"'

        return await self._call_gemini_api_async(translation_prompt)

    def _get_nllb_lang_code(self, lang: str) -> str:
        """Maps simple language codes to NLLB model's required format."""
        mapping = {
            'en': 'eng_Latn', 'hi': 'hin_Deva', 'bn': 'ben_Beng', 'ta': 'tam_Taml',
            'te': 'tel_Telu', 'mr': 'mar_Deva', 'gu': 'guj_Gujr', 'pa': 'pan_Guru',
            'es': 'spa_Latn', 'fr': 'fra_Latn', 'de': 'deu_Latn', 'ar': 'ara_Arab'
        }
        return mapping.get(lang, 'eng_Latn')  # Default to English if not found

    async def _translate_with_nllb(self, text: str, target_lang: str) -> str:
        """Translates literal text using the local NLLB model."""
        print("➡️ Routing to local NLLB Model for direct translation...")
        try:
            source_lang = detect(text)
        except Exception:
            source_lang = 'en'  # Default to English if detection fails

        src_code = self._get_nllb_lang_code(source_lang)
        tgt_code = self._get_nllb_lang_code(target_lang)

        def _translate_sync():
            translator = pipeline("translation", model=self.nllb_model, tokenizer=self.nllb_tokenizer,
                                  src_lang=src_code, tgt_lang=tgt_code, device=self.device_id)
            result = translator(text, max_length=512)
            return result[0]['translation_text']

        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return await loop.run_in_executor(pool, _translate_sync)

    # --- Main Public Method (The Router) ---
    async def process_text(self, input_text: str, target_language: str = "hi") -> str:
        """Processes text by routing it to the appropriate translation model."""
        if not input_text or not isinstance(input_text, str):
            return "Error: Invalid input text provided."

        print(f"\n--- Processing Text: '{input_text}' ---")
        detection_result = await self._detect_figurative_speech(input_text)

        if detection_result == 1:
            print("✅ Figurative speech detected. Routing to Gemini for nuanced translation.")
            return await self._handle_figurative_translation(input_text, target_language)
        else:
            print("❌ No figurative speech detected. Routing to local NLLB model for direct translation.")
            return await self._translate_with_nllb(input_text, target_language)
