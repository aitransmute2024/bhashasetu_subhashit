import os
import json
import time
import re
import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Read the API key from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

KNOWLEDGE_BASE_DB_FILE = r'C:\Users\admin\OneDrive - Aidwise Private Ltd\BhashaSetu_VAM\data\knowledge_base.json'


# --- Figurative Speech Knowledge Base ---
def load_figurative_speech_db() -> list[dict]:
    """Loads figurative speech DB from a JSON file."""
    if os.path.exists(KNOWLEDGE_BASE_DB_FILE):
        with open(KNOWLEDGE_BASE_DB_FILE, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                print("⚠️ Warning: Could not decode JSON. Returning empty DB.")
                return []
    return []


def save_figurative_speech_db(db: list[dict]):
    """Saves figurative speech DB to a JSON file."""
    with open(KNOWLEDGE_BASE_DB_FILE, 'w', encoding='utf-8') as f:
        json.dump(db, f, indent=4, ensure_ascii=False)


# --- Gemini API Helper ---
def call_gemini_api(prompt: str, max_retries: int = 5, initial_delay: int = 1) -> str:
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}

    for attempt in range(max_retries):
        try:
            payload = {
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.0}
            }
            response = requests.post(api_url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()
            result = response.json()

            if result.get('candidates') and result['candidates'][0].get('content') and \
                    result['candidates'][0]['content'].get('parts') and \
                    result['candidates'][0]['content']['parts'][0].get('text'):
                return result['candidates'][0]['content']['parts'][0]['text'].strip()
            else:
                print(f"⚠️ Attempt {attempt + 1}: Unexpected response: {result}")
                time.sleep(initial_delay * (2 ** attempt))
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Attempt {attempt + 1}: Request failed: {e}")
            time.sleep(initial_delay * (2 ** attempt))
        except json.JSONDecodeError as e:
            print(f"⚠️ Attempt {attempt + 1}: JSON decoding failed: {e}")
            time.sleep(initial_delay * (2 ** attempt))
    return "Error"


# --- Figurative Speech Detection ---
def detect_figurative_speech(text: str) -> str:
    db = load_figurative_speech_db()
    text_lower = text.lower()

    # Check local DB first
    for entry in db:
        if entry["figurative_speech"].lower() in text_lower:
            print(f"✅ Found in DB: '{entry['figurative_speech']}'")
            return "Yes"

    # Fallback to Gemini
    prompt = f"""Does the following sentence contain figurative speech (idiom, metaphor, simile, etc.)? 
Respond 'Yes' or 'No' only.

Sentence: "{text}" """
    print("🤖 Checking Gemini API...")
    return call_gemini_api(prompt)


# --- Rewrite Figurative Speech & Update DB ---
def rewrite_figurative_speech(statement: str) -> str:
    db = load_figurative_speech_db()
    statement_lower = statement.lower()

    # Check DB first
    for entry in db:
        if entry["figurative_speech"].lower() in statement_lower:
            print(f"✏️ Rewriting using DB: {entry['figurative_speech']}")
            return statement.replace(entry["figurative_speech"], entry["literal_meaning"])

    # Rewrite via Gemini
    prompt = f"""Rewrite the following sentence by replacing figurative speech with its general literal meaning. 
    Return only one rewritten sentence that sounds natural and grammatically correct.
    Sentence: "{statement}" """

    rewritten = call_gemini_api(prompt)

    # Identify figurative phrase & meaning
    identify_prompt = f"""Identify figurative speech phrase in this sentence and its literal general meaning.
Respond ONLY as valid JSON in this format:
{{
  "figurative_speech": "...",
  "type": "...",
  "literal_meaning": "..."
}}
Sentence: "{statement}" """

    identified_str = call_gemini_api(identify_prompt)
    print(f"🔍 Gemini raw identify response: {identified_str}")

    # Clean Gemini's response
    cleaned_json_str = identified_str.strip()
    if cleaned_json_str.startswith("```"):
        cleaned_json_str = re.sub(r"```(?:json)?", "", cleaned_json_str).strip()

    # Try parsing JSON
    try:
        identified_data = json.loads(cleaned_json_str)
    except json.JSONDecodeError:
        print("⚠️ JSON parsing failed. Attempting regex extraction...")
        identified_data = {}
        match = re.search(r'"figurative_speech"\s*:\s*"([^"]+)"', identified_str)
        if match:
            identified_data["figurative_speech"] = match.group(1)
        match = re.search(r'"literal_meaning"\s*:\s*"([^"]+)"', identified_str)
        if match:
            identified_data["literal_meaning"] = match.group(1)
        match = re.search(r'"type"\s*:\s*"([^"]+)"', identified_str)
        if match:
            identified_data["type"] = match.group(1)

    # Save if valid
    figurative_phrase = identified_data.get("figurative_speech")
    literal_meaning = identified_data.get("literal_meaning")
    if figurative_phrase and literal_meaning:
        new_entry = {
            "figurative_speech": figurative_phrase,
            "type": identified_data.get("type", "N/A"),
            "literal_meaning": literal_meaning
        }
        if not any(e["figurative_speech"].lower() == figurative_phrase.lower() for e in db):
            db.append(new_entry)
            save_figurative_speech_db(db)
            print(f"✅ Added to DB: {figurative_phrase} → {literal_meaning}")
    else:
        print("⚠️ Could not extract figurative speech details.")

    return rewritten


# --- Main Process ---
def process_text(input_text: str) -> str:
    if not input_text:
        return "Error: No input text."

    print(f"\n--- Processing text ---")
    has_figurative_speech = detect_figurative_speech(input_text)

    if has_figurative_speech == "Yes":
        print("Figurative speech detected. Rewriting...")
        rewritten_text = rewrite_figurative_speech(input_text)
        return rewritten_text
    elif has_figurative_speech == "No":
        print("No figurative speech detected.")
        return input_text
    else:
        print("Error detecting figurative speech.")
        return input_text


# --- Example ---
# if __name__ == "__main__":
#     input_text = "He blew his top when he heard the news."
#     result = process_text(input_text)
#     print("\n✅ Final Output:", result)
