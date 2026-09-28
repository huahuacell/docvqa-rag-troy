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