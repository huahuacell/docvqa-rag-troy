import argparse
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.data_processor import DataProcessor
from src.text_retriever import TextRetriever
from src.generator import Generator
from src.evaluator import Evaluator


INDEX_PATH = Path("results/text.index")
METADATA_PATH = Path("results/text_metadata.json")
RECALL_PATH = Path("results/text_document_recall.json")
PREDICTION_PATH = Path("results/text_predictions.json")

TOP_K = 5
MAX_WORKERS = 5


def get_processor():
    return DataProcessor(
        annotation_path="data/annotations/infographicsVQA_val_v1.0_withQT.json",
        image_dir="data/images",
        chunk_dir="data/chunks",
        ocr_dir="data/ocr",
    )


def evaluate_document_recall(force=False):
    if not force and RECALL_PATH.exists():
        print("[SKIP] Text document recall already exists.")

        with open(RECALL_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        print(json.dumps(metrics, indent=2))
        return

    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Text index not found: {INDEX_PATH}"
        )

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Text metadata not found: {METADATA_PATH}"
        )

    processor = get_processor()

    retriever = TextRetriever(
        index_path=INDEX_PATH,
        metadata_path=METADATA_PATH,
    )

    evaluator = Evaluator(
        retriever.metadata
    )

    metrics = evaluator.evaluate_document_recall(
        samples=processor.get_samples(),
        retriever=retriever,
        k=TOP_K,
    )

    RECALL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        RECALL_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metrics,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n===== Text Document Recall =====")
    print(json.dumps(metrics, indent=2))
    print(f"Saved to: {RECALL_PATH}")


def save_predictions(predictions):
    PREDICTION_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        PREDICTION_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            predictions,
            f,
            ensure_ascii=False,
            indent=2,
        )


def generate_predictions():
    processor = get_processor()
    samples = processor.get_samples()

    retriever = TextRetriever(
        index_path=INDEX_PATH,
        metadata_path=METADATA_PATH,
    )

    generator = Generator(
        model="ecnu-plus"
    )

    if PREDICTION_PATH.exists():
        with open(
            PREDICTION_PATH,
            "r",
            encoding="utf-8",
        ) as f:
            predictions = json.load(f)
    else:
        predictions = {}

    remaining = [
        sample
        for sample in samples
        if str(sample["questionId"]) not in predictions
    ]

    print(f"Total questions: {len(samples)}")
    print(f"Already completed: {len(predictions)}")
    print(f"Remaining: {len(remaining)}")
    print(f"Top-K: {TOP_K}")
    print(f"Concurrency: {MAX_WORKERS}")

    if not remaining:
        print("[SKIP] All text predictions already completed.")
        return

    def process(sample):
        qid = str(sample["questionId"])
        question = sample["question"]

        contexts = retriever.retrieve(
            question,
            top_k=TOP_K,
        )

        answer = generator.generate_text(
            question,
            contexts,
        )

        return qid, {
            "questionId": sample["questionId"],
            "question": question,
            "answer": answer,
        }

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        futures = {
            executor.submit(
                process,
                sample,
            ): sample
            for sample in remaining
        }

        for i, future in enumerate(
            as_completed(futures),
            start=1,
        ):
            sample = futures[future]
            qid = str(sample["questionId"])

            try:
                key, result = future.result()
                predictions[key] = result

                print(
                    f"[{len(predictions)}/{len(samples)}] "
                    f"{key}: {result['answer']}"
                )

            except Exception as e:
                print(
                    f"[ERROR] {qid}: {e}"
                )

            if i % 10 == 0:
                save_predictions(predictions)

    save_predictions(predictions)

    print("\nGeneration finished.")
    print(
        f"Predictions: "
        f"{len(predictions)}/{len(samples)}"
    )
    print(
        f"Saved to: {PREDICTION_PATH}"
    )


def run_all(force=False):
    print("\n=== Stage 1: Text Document Recall@5 ===")
    evaluate_document_recall(
        force=force
    )

    print("\n=== Stage 2: Text Generation ===")
    generate_predictions()

    print("\n=== Text RAG Finished ===")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=[
            "all",
            "recall",
            "generate",
        ],
        default="all",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute completed recall result.",
    )

    args = parser.parse_args()

    if args.mode == "all":
        run_all(
            force=args.force
        )

    elif args.mode == "recall":
        evaluate_document_recall(
            force=args.force
        )

    elif args.mode == "generate":
        generate_predictions()


if __name__ == "__main__":
    main()