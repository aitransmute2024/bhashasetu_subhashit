import os
from indicnlp import common
from indicnlp.tokenize.sentence_tokenize import sentence_split


def split_translated_text_with_indicnlp(translated_text, lang_code="hi"):
    # Automatically set path to ../indic/indic_nlp_resources
    script_dir = os.path.dirname(os.path.abspath(__file__))
    indic_resource_path = os.path.abspath(os.path.join(script_dir, "..", "..", "indic", "indic_nlp_resources"))

    # Set the resource path globally
    common.set_resources_path(indic_resource_path)

    # Perform sentence splitting
    return sentence_split(translated_text.strip(), lang_code)

if __name__ == "__main__":
    translated_text = "नमस्ते, आप कैसे हैं?  आज आप क्या कर रहे हैं।"
    lang_code = "en"

    segments = split_translated_text_with_indicnlp(translated_text, lang_code)

    print("🪄 Translated Segments:")
    for i, s in enumerate(segments, 1):
        print(f"{i}. {s}")
