import json
from simalign import SentenceAligner
from sentence_transformers import SentenceTransformer, util
from models.align_models import aligner, embed_model


SIMILARITY_THRESHOLD = 0.5


def align_sentences(src_text, tgt_text, threshold=SIMILARITY_THRESHOLD):
    """
    Aligns words between source and target sentences based on both alignment model and embedding similarity.

    Args:
        src_text (str): Source sentence (e.g., in Hindi).
        tgt_text (str): Target sentence (e.g., in English).
        threshold (float): Cosine similarity threshold for valid alignment.

    Returns:
        dict: JSON-style dictionary with alignment information.
    """
    src_tokens = src_text.strip().split()
    tgt_tokens = tgt_text.strip().split()

    results = aligner.get_word_aligns(src_tokens, tgt_tokens)
    alignment_method = "itermax"
    alignment_indices = results[alignment_method]

    # Get embeddings for all unique words
    all_words = list(set(src_tokens + tgt_tokens))
    embeddings = embed_model.encode(all_words, convert_to_tensor=True)
    embed_dict = {word: embeddings[i] for i, word in enumerate(all_words)}

    aligned_words = []
    for i, j in alignment_indices:
        src_word = src_tokens[i]
        tgt_word = tgt_tokens[j]
        sim_score = util.cos_sim(embed_dict[src_word], embed_dict[tgt_word]).item()
        if sim_score >= threshold:
            aligned_words.append({
                "source": src_word,
                "target": tgt_word,
                "similarity": round(sim_score, 4)
            })

    alignment_json = {
        "alignments": aligned_words
    }

    return alignment_json

#
# # Example usage
# if __name__ == "__main__":
#     src = "मुझे खाना पसंद"
#     tgt = "We don't like food"
#     result = align_sentences(src, tgt)
#     print(json.dumps(result, indent=2, ensure_ascii=False))
