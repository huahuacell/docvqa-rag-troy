import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.data_processor import DataProcessor
from src.text_retriever import TextRetriever
from src.generator import Generator


OUTPUT_PATH = Path("results/text_predictions.json")
MAX_WORKERS = 5


def main():
    processor = DataProcessor(
        annotation_path="data/annotations/infographicsVQA_val_v1.0_withQT.json",
        image_dir="data/images",
        chunk_dir="data/chunks",
        ocr_dir="data/ocr",
    )

    retriever = TextRetriever(
        index_path="results/text.index",
        metadata_path="results/text_metadata.json",
    )

    generator = Generator()
    samples = processor.get_samples()

    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            predictions = json.load(f)
    else:
        predictions = {}

    def process(sample):
        question_id = str(sample["questionId"])
        question = sample["question"]

        contexts = retriever.retrieve(
            question,
            top_k=5,
        )

        answer = generator.generate_text(
            question,
            contexts,
        )

        return question_id, {
            "questionId": sample["questionId"],
            "question": question,
            "answer": answer,
        }

    remaining = [
        sample for sample in samples
        if str(sample["questionId"]) not in predictions
    ]

    print(f"Total questions: {len(samples)}")
    print(f"Already completed: {len(predictions)}")
    print(f"Remaining: {len(remaining)}")
    print(f"Concurrency: {MAX_WORKERS}")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process, sample): sample
            for sample in remaining
        }

        for i, future in enumerate(as_completed(futures), 1):
            sample = futures[future]
            question_id = str(sample["questionId"])

            try:
                qid, result = future.result()
                predictions[qid] = result
                print(
                    f"[{len(predictions)}/{len(samples)}] "
                    f"{qid}: {result['answer']}"
                )

            except Exception as e:
                print(f"[ERROR] {question_id}: {e}")

            if i % 10 == 0:
                OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
                with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
                    json.dump(
                        predictions,
                        f,
                        ensure_ascii=False,
                        indent=2,
                    )

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(
            predictions,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\nGeneration finished.")
    print(f"Predictions: {len(predictions)}/{len(samples)}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()