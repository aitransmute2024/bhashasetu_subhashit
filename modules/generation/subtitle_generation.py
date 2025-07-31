import math
from datetime import timedelta

def format_time(ms):
    td = timedelta(milliseconds=ms)
    return str(td)[:-3].replace('.', ',').zfill(12)

def generate_srt_entries_from_text(translated_text, start_ms, end_ms, max_words_per_line=6):
    words = translated_text.strip().split()
    total_words = len(words)
    total_duration = end_ms - start_ms

    if total_words == 0 or total_duration <= 0:
        return []

    # Calculate average duration per word in ms
    avg_word_duration = total_duration / total_words

    # Split words into chunks
    chunks = [words[i:i + max_words_per_line] for i in range(0, total_words, max_words_per_line)]

    srt_entries = []
    current_start = start_ms

    for idx, chunk in enumerate(chunks, start=1):
        chunk_word_count = len(chunk)
        chunk_duration = int(chunk_word_count * avg_word_duration)
        current_end = current_start + chunk_duration

        # Handle last chunk to match total duration exactly
        if idx == len(chunks):
            current_end = end_ms

        srt_entry = {
            "index": len(srt_entries) + 1,
            "start": format_time(current_start),
            "end": format_time(current_end),
            "text": ' '.join(chunk)
        }

        srt_entries.append(srt_entry)
        current_start = current_end

    return srt_entries
