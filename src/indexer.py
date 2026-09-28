# SigLIP
from pathlib import Path
import json

import faiss
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, SiglipModel


class Indexer:
    def __init__(
        self,
        model_name="google/siglip-base-patch16-224",
        device=None,
    ):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print(f"SigLIP device: {self.device}")

        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = (
            SiglipModel
            .from_pretrained(model_name)
            .to(self.device)
            .eval()
        )

    @torch.no_grad()
    def encode_images(self, image_paths, batch_size=16):
        embeddings = []

        for start in range(0, len(image_paths), batch_size):
            paths = image_paths[start:start + batch_size]

            images = [
                Image.open(path).convert("RGB")
                for path in paths
            ]

            inputs = self.processor(
                images=images,
                return_tensors="pt",
            )

            pixel_values = inputs["pixel_values"].to(self.device)

            outputs = self.model.get_image_features(
                pixel_values=pixel_values
            )

            features = outputs.pooler_output

            features = torch.nn.functional.normalize(
                features,
                p=2,
                dim=-1,
            )

            embeddings.append(
                features.cpu().numpy()
            )

        return np.concatenate(
            embeddings,
            axis=0,
        ).astype("float32")

    def build(
        self,
        chunks,
        index_path,
        metadata_path,
        batch_size=16,
    ):
        if not chunks:
            raise ValueError("No visual chunks provided.")

        image_paths = [
            item["chunk_path"]
            for item in chunks
        ]

        embeddings = self.encode_images(
            image_paths,
            batch_size=batch_size,
        )

        index = faiss.IndexFlatIP(
            embeddings.shape[1]
        )

        index.add(embeddings)

        index_path = Path(index_path)
        metadata_path = Path(metadata_path)

        index_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        metadata_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        faiss.write_index(
            index,
            str(index_path),
        )

        with open(
            metadata_path,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                chunks,
                f,
                ensure_ascii=False,
                indent=2,
            )

        print(f"Indexed visual chunks: {index.ntotal}")
        print(f"Embedding dimension: {embeddings.shape[1]}")

        return index