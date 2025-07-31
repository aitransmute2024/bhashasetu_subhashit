from pydub import AudioSegment

def pause_identification(all_segments, audio_path, min_pause_len=0.3):
    """
    Hybrid pause detection: diarization gaps validated with silence detection.
    """
    audio = AudioSegment.from_file(audio_path)
    pause_segments = []
    prev_end = 0.0

    for start, end, _ in all_segments:
        gap = start - prev_end
        if gap >= min_pause_len:
            gap_audio = audio[int(prev_end * 1000):int(start * 1000)]
            if gap_audio.dBFS < -35:  # energy threshold check
                pause_segments.append({"start": round(prev_end, 2), "end": round(start, 2)})
        prev_end = max(prev_end, end)

    return pause_segments

# # -------------------------
# # Pause Identification
# # -------------------------
# def pause_identification(all_segments, audio_duration):
#     """
#     Identify pauses between speaker segments.
#     """
#     pause_segments = []
#     prev_end = 0.0
#
#     for start, end, _ in all_segments:
#         if start > prev_end:
#             pause_segments.append({
#                 "start": round(prev_end, 2),
#                 "end": round(start, 2)
#             })
#         prev_end = max(prev_end, end)
#
#     if prev_end < audio_duration:
#         pause_segments.append({
#             "start": round(prev_end, 2),
#             "end": round(audio_duration, 2)
#         })
#
#     return pause_segments
#
