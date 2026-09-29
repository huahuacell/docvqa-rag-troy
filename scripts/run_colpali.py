import argparse
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.data_processor import DataProcessor
from src.colpali_indexer import ColPaliIndexer
from src.colpali_retriever import ColPaliRetriever
from src.generator import Generator
from src.evaluator import Evaluator


EMBEDDING_DIR = Path(
    "results/colpali_embeddings"
)

METADATA_PATH = Path(
    "results/colpali_metadata.json"
)

RETRIEVAL_PATH = Path(
    "results/colpali_top5.json"
)

RECALL_PATH = Path(
    "results/colpali_recall.json"
)

DETAILS_PATH = Path(
    "results/"
    "colpali_document_recall_details.json"
)

PREDICTION_PATH = Path(
    "results/colpali_predictions.json"
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


def build_index():
    processor = get_processor()

    chunks = (
        processor.build_chunks()
    )

    print(
        f"Total visual chunks: "
        f"{len(chunks)}"
    )

    indexer = ColPaliIndexer(
        device="cuda"
    )

    indexer.build(
        chunks=chunks,
        embedding_dir=EMBEDDING_DIR,
        metadata_path=METADATA_PATH,
        batch_size=2,
    )

    print(
        "[DONE] ColPali index built."
    )


def retrieve_all():
    processor = get_processor()

    samples = processor.get_samples()

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"ColPali metadata not found: "
            f"{METADATA_PATH}"
        )

    if not EMBEDDING_DIR.exists():
        raise FileNotFoundError(
            f"ColPali embeddings not found: "
            f"{EMBEDDING_DIR}"
        )

    retriever = ColPaliRetriever(
        embedding_dir=EMBEDDING_DIR,
        metadata_path=METADATA_PATH,
        device="cuda",
    )

    if RETRIEVAL_PATH.exists():
        with open(
            RETRIEVAL_PATH,
            "r",
            encoding="utf-8",
        ) as f:
            retrievals = json.load(f)
    else:
        retrievals = {}

    remaining = [
        sample
        for sample in samples
        if str(sample["questionId"])
        not in retrievals
    ]

    print(
        f"Retrieval completed: "
        f"{len(retrievals)}"
    )

    print(
        f"Retrieval remaining: "
        f"{len(remaining)}"
    )

    if not remaining:
        print(
            "[SKIP] All retrievals "
            "already completed."
        )
        return

    for i, sample in enumerate(
        remaining,
        start=1,
    ):
        qid = str(
            sample["questionId"]
        )

        results = retriever.retrieve(
            sample["question"],
            top_k=TOP_K,
        )

        retrievals[
            qid
        ] = results

        print(
            f"[{len(retrievals)}/"
            f"{len(samples)}] "
            f"{qid}"
        )

        # 每 5 题保存一次 checkpoint
        if i % 5 == 0:
            save_json(
                RETRIEVAL_PATH,
                retrievals,
            )

    save_json(
        RETRIEVAL_PATH,
        retrievals,
    )

    print(
        "[DONE] ColPali retrieval finished."
    )

    print(
        f"Saved to: {RETRIEVAL_PATH}"
    )


def evaluate_recall(force=False):
    if (
        not force
        and RECALL_PATH.exists()
        and DETAILS_PATH.exists()
    ):
        print(
            "[SKIP] ColPali recall and "
            "details already exist."
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

    processor = get_processor()

    samples = processor.get_samples()

    if not RETRIEVAL_PATH.exists():
        raise FileNotFoundError(
            f"Retrieval cache not found: "
            f"{RETRIEVAL_PATH}"
        )

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata not found: "
            f"{METADATA_PATH}"
        )

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        metadata = json.load(f)

    with open(
        RETRIEVAL_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        retrievals = json.load(f)

    print(
        f"Cached retrievals: "
        f"{len(retrievals)}"
    )

    if len(retrievals) != len(samples):
        print(
            "[WARNING] Retrieval cache "
            "does not contain all questions."
        )

        print(
            f"Expected: {len(samples)}, "
            f"found: {len(retrievals)}"
        )

    evaluator = Evaluator(
        metadata
    )

    metrics, details = (
        evaluator
        .evaluate_document_recall_from_cache(
            samples=samples,
            retrievals=retrievals,
            k=TOP_K,
        )
    )

    save_json(
        RECALL_PATH,
        metrics,
    )

    save_json(
        DETAILS_PATH,
        details,
    )

    print(
        "\n===== ColPali "
        "Document Recall ====="
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


def generate_predictions():
    processor = get_processor()

    samples = (
        processor.get_samples()
    )

    if not RETRIEVAL_PATH.exists():
        raise FileNotFoundError(
            f"Retrieval cache not found: "
            f"{RETRIEVAL_PATH}"
        )

    with open(
        RETRIEVAL_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        retrievals = json.load(f)

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
        f"Generation completed: "
        f"{len(predictions)}"
    )

    print(
        f"Generation remaining: "
        f"{len(remaining)}"
    )

    print(
        f"Concurrency: "
        f"{MAX_WORKERS}"
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

        if qid not in retrievals:
            raise KeyError(
                f"No retrieval result "
                f"for question {qid}"
            )

        contexts = retrievals[
            qid
        ]

        answer = (
            generator.generate_visual(
                sample["question"],
                contexts,
            )
        )

        return qid, {
            "questionId": (
                sample["questionId"]
            ),
            "question": (
                sample["question"]
            ),
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
            sample = futures[
                future
            ]

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

            # Kaggle 更安全
            if i % 5 == 0:
                save_json(
                    PREDICTION_PATH,
                    predictions,
                )

    save_json(
        PREDICTION_PATH,
        predictions,
    )

    print(
        "[DONE] ColPali "
        "generation finished."
    )

    print(
        f"Predictions: "
        f"{len(predictions)}/"
        f"{len(samples)}"
    )

    print(
        f"Saved to: "
        f"{PREDICTION_PATH}"
    )


def save_json(
    path,
    data,
):
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )


def run_all(force=False):
    print(
        "\n=== 1. ColPali Index ==="
    )

    build_index()

    print(
        "\n=== 2. ColPali Retrieval ==="
    )

    retrieve_all()

    print(
        "\n=== 3. Document Recall@5 ==="
    )

    evaluate_recall(
        force=force
    )

    print(
        "\n=== 4. Generation ==="
    )

    generate_predictions()

    print(
        "\n=== ColPali Finished ==="
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=[
            "all",
            "index",
            "retrieve",
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
            "recall evaluation."
        ),
    )

    args = parser.parse_args()

    if args.mode == "all":
        run_all(
            force=args.force
        )

    elif args.mode == "index":
        build_index()

    elif args.mode == "retrieve":
        retrieve_all()

    elif args.mode == "recall":
        evaluate_recall(
            force=args.force
        )

    elif args.mode == "generate":
        generate_predictions()


if __name__ == "__main__":
    main()