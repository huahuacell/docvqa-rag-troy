import json
from pathlib import Path

import torch
from transformers import (
    ColPaliForRetrieval,
    ColPaliProcessor,
)


class ColPaliRetriever:
    def __init__(
        self,
        embedding_dir,
        metadata_path,
        model_name="vidore/colpali-v1.3-hf",
        device=None,
    ):
        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        if self.device == "cpu":
            raise RuntimeError(
                "ColPali retrieval requires GPU."
            )

        self.embedding_dir = Path(
            embedding_dir
        )

        with open(
            metadata_path,
            "r",
            encoding="utf-8",
        ) as f:
            self.metadata = json.load(f)

        self.processor = (
            ColPaliProcessor
            .from_pretrained(model_name)
        )

        self.processor.query_prefix = "Query: "

        self.model = (
            ColPaliForRetrieval
            .from_pretrained(
                model_name,
                torch_dtype=torch.float16,
            )
            .to(self.device)
            .eval()
        )

        self.batch_files = sorted(
            self.embedding_dir.glob(
                "batch_*.pt"
            )
        )

        if not self.batch_files:
            raise FileNotFoundError(
                "No ColPali embedding batches found."
            )

    @torch.no_grad()
    def encode_query(
        self,
        query,
    ):
        inputs = self.processor(
            text=[query],
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        outputs = self.model(
            **inputs
        )

        return outputs.embeddings

    def retrieve(
        self,
        query,
        top_k=5,
    ):
        query_embeddings = (
            self.encode_query(query)
        )

        candidates = []

        offset = 0

        for batch_file in self.batch_files:
            doc_embeddings = torch.load(
                batch_file,
                map_location="cpu",
            )

            batch_size = (
                doc_embeddings.shape[0]
            )

            docs_gpu = (
                doc_embeddings
                .to(
                    self.device,
                    dtype=torch.float16,
                )
            )

            with torch.no_grad():
                scores = (
                    self.processor
                    .score_retrieval(
                        query_embeddings,
                        docs_gpu,
                    )
                )

            scores = (
                scores[0]
                .float()
                .cpu()
            )

            batch_k = min(
                top_k,
                len(scores),
            )

            values, indices = (
                torch.topk(
                    scores,
                    k=batch_k,
                )
            )

            for score, idx in zip(
                values.tolist(),
                indices.tolist(),
            ):
                candidates.append(
                    (
                        offset + idx,
                        score,
                    )
                )

            offset += batch_size

            del docs_gpu
            del doc_embeddings

        candidates.sort(
            key=lambda x: x[1],
            reverse=True,
        )

        candidates = candidates[
            :top_k
        ]

        results = []

        for rank, (
            idx,
            score,
        ) in enumerate(
            candidates,
            start=1,
        ):
            item = (
                self.metadata[idx]
                .copy()
            )

            item["rank"] = rank
            item["score"] = float(
                score
            )

            results.append(item)

        return results