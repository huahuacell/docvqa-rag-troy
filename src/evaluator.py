import re


class Evaluator:
    def __init__(self, metadata):
        self.metadata = metadata

    @staticmethod
    def normalize(text):
        text = str(text).lower().strip()
        return re.sub(r"\s+", " ", text)

    def get_relevant_chunks(self, sample):
        image_name = sample["image_local_name"]
        answers = [self.normalize(x) for x in sample["answers"]]
        relevant = set()

        for item in self.metadata:
            if item["image_name"] != image_name:
                continue

            text = self.normalize(item["text"])

            if any(answer and answer in text for answer in answers):
                relevant.add(item["chunk_id"])

        return relevant

    def recall_at_k(self, retrieved, relevant_chunks, k=5):
        if not relevant_chunks:
            return None

        retrieved_ids = {
            item["chunk_id"]
            for item in retrieved[:k]
        }

        return int(bool(retrieved_ids & relevant_chunks))

    def evaluate_retrieval(self, samples, retriever, k=5):
        hits = 0
        total = 0

        for i, sample in enumerate(samples, 1):
            relevant = self.get_relevant_chunks(sample)

            if not relevant:
                continue

            retrieved = retriever.retrieve(
                sample["question"],
                top_k=k
            )

            hits += self.recall_at_k(
                retrieved,
                relevant,
                k
            )
            total += 1

            if i % 100 == 0:
                print(f"[{i}/{len(samples)}]")

        return {
            f"recall@{k}": hits / total if total else 0,
            "hits": hits,
            "evaluated_questions": total,
            "total_questions": len(samples),
        }
    @staticmethod
    def point_in_bbox(x, y, bbox):
        x1, y1, x2, y2 = bbox

        return (
            x1 <= x <= x2
            and y1 <= y <= y2
        )

    def get_relevant_visual_chunks(
        self,
        sample,
        processor,
        lines_per_chunk=10,
        overlap_lines=2,
    ):
        image_name = sample["image_local_name"]

        answers = [
            self.normalize(x)
            for x in sample["answers"]
        ]

        # 和 Text RAG 完全一样的 OCR chunk 划分
        ocr_chunks = processor.chunk_ocr(
            image_name,
            lines_per_chunk=lines_per_chunk,
            overlap_lines=overlap_lines,
        )

        # 找包含 GT answer 的 OCR chunks
        evidence_boxes = []

        for chunk in ocr_chunks:
            text = self.normalize(
                chunk["text"]
            )

            if any(
                answer and answer in text
                for answer in answers
            ):
                evidence_boxes.append(
                    chunk["bbox"]
                )

        # 与 Text Recall 一样：
        # 找不到 GT evidence 的题不参与 Recall
        if not evidence_boxes:
            return set()

        image_path = (
            processor.image_dir
            / image_name
        )

        from PIL import Image

        with Image.open(image_path) as image:
            width, height = image.size

        relevant = set()

        # Visual metadata 中 bbox 是 pixel 坐标
        for item in self.metadata:
            if item["image_name"] != image_name:
                continue

            x1, y1, x2, y2 = item["bbox"]

            visual_bbox = [
                x1 / width,
                y1 / height,
                x2 / width,
                y2 / height,
            ]

            # OCR bbox 是 normalized [0,1]
            # 用 evidence bbox 中心判断它落在哪个 visual chunk
            for evidence in evidence_boxes:
                cx = (
                    evidence[0] + evidence[2]
                ) / 2

                cy = (
                    evidence[1] + evidence[3]
                ) / 2

                if self.point_in_bbox(
                    cx,
                    cy,
                    visual_bbox,
                ):
                    relevant.add(
                        item["chunk_id"]
                    )
                    break

        return relevant

    def evaluate_visual_retrieval(
        self,
        samples,
        retriever,
        processor,
        eval_ids,
        k=5,
    ):
        eval_ids = {
            str(qid)
            for qid in eval_ids
        }

        hits = 0
        total = 0
        missing_relevance = 0

        for i, sample in enumerate(samples, 1):
            qid = str(sample["questionId"])

            if qid not in eval_ids:
                continue

            relevant = self.get_relevant_visual_chunks(
                sample,
                processor,
            )

            if not relevant:
                missing_relevance += 1
                continue

            retrieved = retriever.retrieve(
                sample["question"],
                top_k=k,
            )

            hits += self.recall_at_k(
                retrieved,
                relevant,
                k,
            )

            total += 1

            if total % 100 == 0:
                print(
                    f"evaluated={total}, "
                    f"hits={hits}, "
                    f"missing_relevance={missing_relevance}"
                )

        return {
            f"recall@{k}": hits / total if total else 0,
            "hits": hits,
            "evaluated_questions": total,
            "target_questions": len(eval_ids),
            "missing_relevance": missing_relevance,
            "total_questions": len(samples),
        }
    def evaluate_document_recall(
        self,
        samples,
        retriever,
        k=5,
    ):
        hits = 0
        total = len(samples)
        details = {}

        for i, sample in enumerate(samples, 1):
            qid = str(sample["questionId"])
            question = sample["question"]
            target_image = sample["image_local_name"]

            retrieved = retriever.retrieve(
                question,
                top_k=k,
            )

            retrieved_items = []

            hit = False

            for rank, item in enumerate(
                retrieved[:k],
                start=1,
            ):
                image_name = item["image_name"]

                if image_name == target_image:
                    hit = True

                retrieved_item = {
                    "rank": rank,
                    "image_name": image_name,
                }

                if "chunk_id" in item:
                    retrieved_item["chunk_id"] = item["chunk_id"]

                if "score" in item:
                    retrieved_item["score"] = float(item["score"])

                if "text" in item:
                    retrieved_item["text"] = item["text"]

                if "bbox" in item:
                    retrieved_item["bbox"] = item["bbox"]

                retrieved_items.append(
                    retrieved_item
                )

            hits += int(hit)

            details[qid] = {
                "questionId": sample["questionId"],
                "question": question,
                "target_image": target_image,
                "hit": hit,
                "retrieved": retrieved_items,
            }

            if i % 100 == 0:
                print(
                    f"[{i}/{total}] "
                    f"hits={hits}"
                )

        metrics = {
            f"document_recall@{k}": (
                hits / total
                if total
                else 0
            ),
            "hits": hits,
            "evaluated_questions": total,
        }

        return metrics, details