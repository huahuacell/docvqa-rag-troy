# support 方案 B：Image → visual chunks 方案 A：OCR JSON → text chunks

from pathlib import Path
import json
from PIL import Image


class DataProcessor:
    def __init__(
        self,
        annotation_path,
        image_dir,
        chunk_dir,
        ocr_dir="data/ocr",
        chunk_size=768,
        overlap=128,
    ):
        self.annotation_path = Path(annotation_path)
        self.image_dir = Path(image_dir)
        self.chunk_dir = Path(chunk_dir)
        self.ocr_dir = Path(ocr_dir)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunk_dir.mkdir(parents=True, exist_ok=True)

        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")

        with open(self.annotation_path, "r", encoding="utf-8") as f:
            self.dataset = json.load(f)

        if self.dataset.get("dataset_split") != "val":
            raise ValueError(f"Expected val split, got {self.dataset.get('dataset_split')}")

        self.samples = self.dataset["data"]

    def get_samples(self):
        return self.samples

    def get_unique_images(self):
        return sorted({x["image_local_name"] for x in self.samples})

    def get_unique_ocr_files(self):
        return sorted({x["ocr_output_file"] for x in self.samples})

    def get_ocr_map(self):
        return {
            x["image_local_name"]: x["ocr_output_file"]
            for x in self.samples
        }

    def check_images(self):
        return [
            x for x in self.get_unique_images()
            if not (self.image_dir / x).exists()
        ]

    def check_ocr_files(self):
        return [
            x for x in self.get_unique_ocr_files()
            if not (self.ocr_dir / x).exists()
        ]

    # ---------- Visual RAG ----------

    def chunk_image(self, image_name):
        image_path = self.image_dir / image_name
        if not image_path.exists():
            raise FileNotFoundError(image_path)

        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        size, stride = self.chunk_size, self.chunk_size - self.overlap
        stem = Path(image_name).stem
        chunks = []

        def get_starts(length):
            if length <= size:
                return [0]

            starts = list(range(0, length - size + 1, stride))
            last = length - size

            if starts[-1] != last:
                if last - starts[-1] < stride // 2:
                    starts[-1] = last
                else:
                    starts.append(last)

            return starts

        xs = get_starts(width)
        ys = get_starts(height)

        for y in ys:
            for x in xs:
                x2, y2 = min(x + size, width), min(y + size, height)
                chunk_id = f"{stem}_{len(chunks):04d}"
                chunk_path = self.chunk_dir / f"{chunk_id}.jpg"

                image.crop((x, y, x2, y2)).save(
                    chunk_path, "JPEG", quality=95
                )

                chunks.append({
                    "chunk_id": chunk_id,
                    "image_name": image_name,
                    "chunk_path": str(chunk_path),
                    "bbox": [x, y, x2, y2],
                })

        return chunks

    def build_chunks(self):
        chunks = []
        images = self.get_unique_images()

        for i, image_name in enumerate(images, 1):
            print(f"[{i}/{len(images)}] {image_name}")
            chunks.extend(self.chunk_image(image_name))

        return chunks

    # ---------- Text RAG ----------

    def load_ocr_lines(self, image_name):
        ocr_map = self.get_ocr_map()

        if image_name not in ocr_map:
            raise KeyError(f"No OCR file mapped to {image_name}")

        ocr_path = self.ocr_dir / ocr_map[image_name]
        if not ocr_path.exists():
            raise FileNotFoundError(ocr_path)

        with open(ocr_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        lines = []

        for item in data.get("LINE", []):
            text = item.get("Text", "").strip()
            if not text:
                continue

            box = item.get("Geometry", {}).get("BoundingBox", {})
            left = float(box.get("Left", 0))
            top = float(box.get("Top", 0))
            width = float(box.get("Width", 0))
            height = float(box.get("Height", 0))

            lines.append({
                "text": text,
                "confidence": float(item.get("Confidence", 0)),
                "bbox": [
                    left,
                    top,
                    left + width,
                    top + height,
                ],
            })

        lines.sort(key=lambda x: (x["bbox"][1], x["bbox"][0]))
        return lines

    @staticmethod
    def merge_bboxes(lines):
        boxes = [x["bbox"] for x in lines]

        return [
            min(x[0] for x in boxes),
            min(x[1] for x in boxes),
            max(x[2] for x in boxes),
            max(x[3] for x in boxes),
        ]

    def chunk_ocr(self, image_name, lines_per_chunk=10, overlap_lines=2):
        if overlap_lines >= lines_per_chunk:
            raise ValueError("overlap_lines must be smaller than lines_per_chunk")

        lines = self.load_ocr_lines(image_name)
        step = lines_per_chunk - overlap_lines
        stem = Path(image_name).stem
        chunks = []

        for start in range(0, len(lines), step):
            selected = lines[start:start + lines_per_chunk]

            if not selected:
                continue

            chunk_id = f"{stem}_text_{len(chunks):04d}"

            chunks.append({
                "chunk_id": chunk_id,
                "image_name": image_name,
                "text": " ".join(x["text"] for x in selected),
                "bbox": self.merge_bboxes(selected),
                "line_start": start,
                "line_end": start + len(selected) - 1,
                "avg_confidence": sum(
                    x["confidence"] for x in selected
                ) / len(selected),
            })

            if start + lines_per_chunk >= len(lines):
                break

        return chunks

    def build_text_chunks(self, lines_per_chunk=10, overlap_lines=2):
        chunks = []
        images = self.get_unique_images()

        for i, image_name in enumerate(images, 1):
            print(f"[{i}/{len(images)}] OCR {image_name}")
            chunks.extend(
                self.chunk_ocr(
                    image_name,
                    lines_per_chunk=lines_per_chunk,
                    overlap_lines=overlap_lines,
                )
            )

        return chunks