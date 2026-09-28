import json
import faiss
from sentence_transformers import SentenceTransformer


class TextRetriever:
    def __init__(
        self,
        index_path,
        metadata_path,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        device=None,
    ):
        self.index = faiss.read_index(str(index_path))
        self.model = SentenceTransformer(model_name, device=device)

        with open(metadata_path, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

    def encode_query(self, query):
        return self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")

    def retrieve(self, query, top_k=5):
        query_embedding = self.encode_query(query)
        scores, indices = self.index.search(query_embedding, top_k)

        results = []

        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), 1):
            if idx < 0:
                continue

            item = self.metadata[idx].copy()
            item["rank"] = rank
            item["score"] = float(score)
            results.append(item)

        return results