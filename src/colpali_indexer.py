import json
from pathlib import Path

import torch
from PIL import Image
from transformers import (
    ColPaliForRetrieval,
    ColPaliProcessor,
)


class ColPaliIndexer:
    def __init__(
        self,
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
                "ColPali requires GPU for this experiment."
            )

        print(
            f"ColPali device: {self.device}"
        )

        self.processor = (
            ColPaliProcessor
            .from_pretrained(model_name)
        )

        # This checkpoint was trained with this prefix.
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

    @torch.no_grad()
    def encode_batch(
        self,
        image_paths,
    ):
        images = []

        for path in image_paths:
            with Image.open(path) as image:
                images.append(
                    image.convert("RGB")
                )

        inputs = self.processor(
            images=images,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        outputs = self.model(
            **inputs
        )

        return (
            outputs.embeddings
            .detach()
            .to(torch.float16)
            .cpu()
        )

    def build(
        self,
        chunks,
        embedding_dir,
        metadata_path,
        batch_size=2,
    ):
        if not chunks:
            raise ValueError(
                "No visual chunks provided."
            )

        embedding_dir = Path(
            embedding_dir
        )

        metadata_path = Path(
            metadata_path
        )

        embedding_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        metadata_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        for start in range(
            0,
            len(chunks),
            batch_size,
        ):
            end = min(
                start + batch_size,
                len(chunks),
            )

            batch_id = (
                start // batch_size
            )

            output_path = (
                embedding_dir
                / f"batch_{batch_id:05d}.pt"
            )

            if output_path.exists():
                print(
                    f"[SKIP] "
                    f"{start}:{end}"
                )
                continue

            batch_chunks = chunks[
                start:end
            ]

            paths = [
                item["chunk_path"]
                for item in batch_chunks
            ]

            embeddings = (
                self.encode_batch(paths)
            )

            torch.save(
                embeddings,
                output_path,
            )

            print(
                f"[{end}/{len(chunks)}] "
                f"{output_path.name}"
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

        print(
            "[DONE] ColPali indexing finished."
        )