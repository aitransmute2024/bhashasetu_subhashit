import asyncio
import concurrent.futures
from langdetect import detect
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import argostranslate.package
import argostranslate.translate
from googletrans import Translator

# --- Setup Translation Models ---
nllb_tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
nllb_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
translator_google = Translator()

# --- Language mapping for NLLB ---
NLLB_LANG_CODES = {
    'en': 'eng_Latn', 'hi': 'hin_Deva', 'bn': 'ben_Beng', 'ta': 'tam_Taml',
    'te': 'tel_Telu', 'ml': 'mal_Mlym', 'kn': 'kan_Knda', 'gu': 'guj_Gujr',
    'mr': 'mar_Deva', 'pa': 'pan_Guru', 'ur': 'urd_Arab', 'as': 'asm_Beng',
    'or': 'ory_Orya', 'ne': 'nep_Deva', 'fr': 'fra_Latn', 'de': 'deu_Latn',
    'es': 'spa_Latn'
}


def get_nllb_lang_code(lang):
    return NLLB_LANG_CODES.get(lang, 'eng_Latn')


# --- Function: Detect Language + Translate ---
async def detect_and_translate(text: str, target_lang: str = 'en', backend: str = 'nllb'):
    """
    Detects language of the given text and translates it to target_lang.

    :param text: Source text
    :param target_lang: Target language code (e.g., 'en', 'hi', 'fr')
    :param backend: Translation backend ('nllb', 'google', 'argos')
    :return: dict with detected language and translated text
    """
    loop = asyncio.get_event_loop()

    # Detect language
    try:
        detected_lang = detect(text)
    except Exception:
        detected_lang = "unknown"

    # Translation
    if backend == "google":
        with concurrent.futures.ThreadPoolExecutor() as pool:
            translated_text = await loop.run_in_executor(
                pool,
                lambda: translator_google.translate(text, dest=target_lang, src=detected_lang).text
            )

    elif backend == "nllb":
        src = get_nllb_lang_code(detected_lang)
        tgt = get_nllb_lang_code(target_lang)

        def _translate():
            translator = pipeline("translation", model=nllb_model, tokenizer=nllb_tokenizer, src_lang=src, tgt_lang=tgt)
            result = translator(text, max_length=512)
            return result[0]['translation_text']

        with concurrent.futures.ThreadPoolExecutor() as pool:
            translated_text = await loop.run_in_executor(pool, _translate)

    elif backend == "argos":
        argostranslate.package.update_package_index()
        available_packages = argostranslate.package.get_available_packages()

        def _translate():
            package = next(
                (pkg for pkg in available_packages if pkg.from_code == detected_lang and pkg.to_code == target_lang),
                None)
            if package and not package.is_installed():
                argostranslate.package.install_from_path(package.download())
            return argostranslate.translate.translate(text, detected_lang, target_lang)

        with concurrent.futures.ThreadPoolExecutor() as pool:
            translated_text = await loop.run_in_executor(pool, _translate)

    else:
        translated_text = text  # fallback

    return {
        "detected_language": detected_lang,
        "translated_text": translated_text
    }
