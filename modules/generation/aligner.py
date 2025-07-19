import json
from sentence_transformers import util
from models.align_models import aligner, embed_model

SIMILARITY_THRESHOLD = 0.5
MAX_SPAN_LEN = 4  # Max words in a target span


def align_sentences(source_segments, target_text, threshold=SIMILARITY_THRESHOLD):
    """
    Aligns source segments to spans of the target sentence, maintaining serial order and using embedding similarity.

    Args:
        source_segments (List[str]): List of source segments (1 to many words).
        target_text (str): Target sentence (entire English sentence).
        threshold (float): Cosine similarity threshold to consider a valid alignment.

    Returns:
        dict: JSON-style dictionary of alignments.
    """
    if not source_segments or not target_text.strip():
        return {"alignments": []}

    tgt_tokens = target_text.strip().split()
    if not tgt_tokens:
        return {"alignments": []}

    used_indices = set()
    alignments = []

    # Precompute target embeddings
    tgt_embeddings = embed_model.encode(tgt_tokens, convert_to_tensor=True)

    for src_segment in source_segments:
        src_tokens = src_segment.strip().split()
        if not src_tokens:
            continue

        src_embed = embed_model.encode(src_tokens, convert_to_tensor=True).mean(dim=0)

        best_score = -1
        best_span = None

        for start in range(len(tgt_tokens)):
            for end in range(start + 1, min(start + MAX_SPAN_LEN + 1, len(tgt_tokens) + 1)):
                if any(i in used_indices for i in range(start, end)):
                    continue  # skip if overlap with previous matches

                span_embed = tgt_embeddings[start:end].mean(dim=0)
                sim_score = util.cos_sim(src_embed, span_embed).item()

                if sim_score > best_score:
                    best_score = sim_score
                    best_span = (start, end)

        if best_score >= threshold and best_span:
            span_start, span_end = best_span
            matched_target = " ".join(tgt_tokens[span_start:span_end])
            alignments.append({
                "source": src_segment,
                "target": matched_target,
                "similarity": round(best_score, 4)
            })
            used_indices.update(range(span_start, span_end))
    print(alignments)
    return {"alignments": alignments}


# source_segments = [
#     "मुझे चाय",
#     "सुबह पीना",
#     "बहुत पसंद है"
# ]
#
# target_text = "I really like to drink tea in the morning"
#
# result = align_segments_to_target(source_segments, target_text)
# print(json.dumps(result, indent=2, ensure_ascii=False))
