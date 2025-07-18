from simalign import SentenceAligner
from sentence_transformers import SentenceTransformer

# Load Simalign aligner with XLM-Roberta
aligner = SentenceAligner(model="xlm-roberta-base", token_type="bpe")

# Load SentenceTransformer model
embed_model = SentenceTransformer("sentence-transformers/LaBSE")
