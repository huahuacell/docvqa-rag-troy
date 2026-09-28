import json

import faiss
import torch
from transformers import AutoProcessor, SiglipModel


class Retriever:
    def __init__(
        self,
        index_path,
        metadata_path,
        model_name="google/siglip-base-patch16-224",
        device=None,
    ):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print(f"SigLIP retriever device: {self.device}")

        self.index = faiss.read_index(
            str(index_path)
        )

        with open(
            metadata_path,
            "r",
            encoding="utf-8",
        ) as f:
            self.metadata = json.load(f)

        self.processor = AutoProcessor.from_pretrained(
            model_name
        )

        self.model = (
            SiglipModel
            .from_pretrained(model_name)
            .to(self.device)
            .eval()
        )

    @torch.no_grad()
    def encode_query(self, query):
        inputs = self.processor(
            text=[query],
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(
            self.device
        )

        outputs = self.model.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        features = outputs.pooler_output

        features = torch.nn.functional.normalize(
            features,
            p=2,
            dim=-1,
        )

        return features.cpu().numpy().astype("float32")

    def retrieve(self, query, top_k=5):
        embedding = self.encode_query(query)

        scores, indices = self.index.search(
            embedding,
            top_k,
        )

        results = []

        for rank, (idx, score) in enumerate(
            zip(indices[0], scores[0]),
            start=1,
        ):
            if idx < 0:
                continue

            item = self.metadata[idx].copy()
            item["rank"] = rank
            item["score"] = float(score)

            results.append(item)

        return results