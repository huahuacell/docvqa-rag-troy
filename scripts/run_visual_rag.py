import argparse
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.data_processor import DataProcessor
from src.indexer import Indexer
from src.retriever import Retriever
from src.generator import Generator
from src.evaluator import Evaluator


INDEX_PATH = Path("results/siglip.index")
METADATA_PATH = Path("results/siglip_metadata.json")

RECALL_PATH = Path(
    "results/siglip_recall.json"
)

DETAILS_PATH = Path(
    "results/siglip_document_recall_details.json"
)

PREDICTION_PATH = Path(
    "results/siglip_predictions.json"
)

TOP_K = 5
MAX_WORKERS = 5


def get_processor():
    return DataProcessor(
        annotation_path=(
            "data/annotations/"
            "infographicsVQA_val_v1.0_withQT.json"
        ),
        image_dir="data/images",
        chunk_dir="data/chunks",
        ocr_dir="data/ocr",
    )


def build_index(force=False):
    if (
        not force
        and INDEX_PATH.exists()
        and METADATA_PATH.exists()
    ):
        print(
            "[SKIP] SigLIP index already exists."
        )
        return

    processor = get_processor()

    chunks = processor.build_chunks()

    print(
        f"Total visual chunks: {len(chunks)}"
    )

    indexer = Indexer(
        # device="cuda"
        device="cpu"
    )

    indexer.build(
        chunks=chunks,
        index_path=INDEX_PATH,
        metadata_path=METADATA_PATH,
        batch_size=16,
    )

    print(
        "[DONE] SigLIP index built."
    )


def evaluate_recall(force=False):
    if (
        not force
        and RECALL_PATH.exists()
        and DETAILS_PATH.exists()
    ):
        print(
            "[SKIP] SigLIP recall and details "
            "already exist."
        )

        with open(
            RECALL_PATH,
            "r",
            encoding="utf-8",
        ) as f:
            metrics = json.load(f)

        print(
            json.dumps(
                metrics,
                indent=2,
            )
        )

        return

    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"SigLIP index not found: "
            f"{INDEX_PATH}"
        )

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"SigLIP metadata not found: "
            f"{METADATA_PATH}"
        )

    processor = get_processor()

    retriever = Retriever(
        index_path=INDEX_PATH,
        metadata_path=METADATA_PATH,
        # device="cuda",
        device="cpu",
    )

    evaluator = Evaluator(
        retriever.metadata
    )

    metrics, details = (
        evaluator.evaluate_document_recall(
            samples=processor.get_samples(),
            retriever=retriever,
            k=TOP_K,
        )
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

    with open(
        DETAILS_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            details,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        "\n===== SigLIP Document Recall ====="
    )

    print(
        json.dumps(
            metrics,
            indent=2,
        )
    )

    print(
        f"Saved metrics to: "
        f"{RECALL_PATH}"
    )

    print(
        f"Saved details to: "
        f"{DETAILS_PATH}"
    )

    print(
        "[DONE] Recall evaluation finished."
    )


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

    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"SigLIP index not found: "
            f"{INDEX_PATH}"
        )

    retriever = Retriever(
        index_path=INDEX_PATH,
        metadata_path=METADATA_PATH,
        # device="cuda",
        device="cpu",
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
        if str(sample["questionId"])
        not in predictions
    ]

    print(
        f"Total: {len(samples)}"
    )

    print(
        f"Completed: {len(predictions)}"
    )

    print(
        f"Remaining: {len(remaining)}"
    )

    print(
        f"Concurrency: {MAX_WORKERS}"
    )

    if not remaining:
        print(
            "[SKIP] All predictions "
            "already completed."
        )
        return

    def process(sample):
        qid = str(
            sample["questionId"]
        )

        question = sample["question"]

        contexts = retriever.retrieve(
            question,
            top_k=TOP_K,
        )

        answer = (
            generator.generate_visual(
                question,
                contexts,
            )
        )

        return qid, {
            "questionId": (
                sample["questionId"]
            ),
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

            qid = str(
                sample["questionId"]
            )

            try:
                key, result = (
                    future.result()
                )

                predictions[
                    key
                ] = result

                print(
                    f"[{len(predictions)}/"
                    f"{len(samples)}] "
                    f"{key}: "
                    f"{result['answer']}"
                )

            except Exception as e:
                print(
                    f"[ERROR] "
                    f"{qid}: {e}"
                )

            # Kaggle 上更频繁保存
            if i % 5 == 0:
                save_predictions(
                    predictions
                )

    save_predictions(
        predictions
    )

    print(
        f"[DONE] Predictions: "
        f"{len(predictions)}/"
        f"{len(samples)}"
    )

    print(
        f"Saved to: "
        f"{PREDICTION_PATH}"
    )


def run_all(force=False):
    print(
        "\n=== Stage 1: SigLIP Index ==="
    )

    build_index(
        force=force
    )

    print(
        "\n=== Stage 2: "
        "Document Recall@5 ==="
    )

    evaluate_recall(
        force=force
    )

    print(
        "\n=== Stage 3: Generation ==="
    )

    generate_predictions()

    print(
        "\n=== Baseline 2 Finished ==="
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=[
            "all",
            "index",
            "recall",
            "generate",
        ],
        default="all",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Recompute completed "
            "index/recall stages."
        ),
    )

    args = parser.parse_args()

    if args.mode == "all":
        run_all(
            force=args.force
        )

    elif args.mode == "index":
        build_index(
            force=args.force
        )

    elif args.mode == "recall":
        evaluate_recall(
            force=args.force
        )

    elif args.mode == "generate":
        generate_predictions()


if __name__ == "__main__":
    main()