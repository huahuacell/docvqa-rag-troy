# official OCR
# Text RAG baseline uses the official Amazon Textract OCR annotations released with InfographicsVQA
from pathlib import Path
import json
import faiss
from sentence_transformers import SentenceTransformer


class TextIndexer:
    def __init__(
        self,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device=None,
    ):
        self.model = SentenceTransformer(model_name, device=device)

    def encode_texts(self, texts, batch_size=64):
        return self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).astype("float32")

    def build(self, chunks, index_path, metadata_path, batch_size=64):
        if not chunks:
            raise ValueError("No text chunks provided.")

        texts = [x["text"] for x in chunks]
        embeddings = self.encode_texts(texts, batch_size)

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        index_path = Path(index_path)
        metadata_path = Path(metadata_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(index, str(index_path))

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

        print(f"Indexed text chunks: {index.ntotal}")
        print(f"Embedding dimension: {embeddings.shape[1]}")

        return index