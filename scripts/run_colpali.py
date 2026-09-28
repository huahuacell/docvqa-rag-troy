import argparse
import json
from pathlib import Path
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)

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

PREDICTION_PATH = Path(
    "results/colpali_predictions.json"
)
EVAL_IDS_PATH = Path(
    "data/annotations/text_recall_eval_ids.json"
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


def retrieve_all():
    processor = get_processor()

    samples = processor.get_samples()

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

        retrievals[qid] = results

        print(
            f"[{len(retrievals)}/"
            f"{len(samples)}] "
            f"{qid}"
        )

        if i % 5 == 0:
            save_json(
                RETRIEVAL_PATH,
                retrievals,
            )

    save_json(
        RETRIEVAL_PATH,
        retrievals,
    )


def evaluate_recall():
    processor = get_processor()
    samples = processor.get_samples()

    with open(
        EVAL_IDS_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        eval_ids = {
            str(qid)
            for qid in json.load(f)
        }

    print(
        f"Fixed evaluation questions: "
        f"{len(eval_ids)}"
    )

    with open(
        METADATA_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        metadata = json.load(f)

    evaluator = Evaluator(
        metadata
    )

    with open(
        RETRIEVAL_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        retrievals = json.load(f)

    hits = 0
    total = 0
    missing_relevance = 0

    for sample in samples:
        qid = str(
            sample["questionId"]
        )

        if qid not in eval_ids:
            continue

        relevant = (
            evaluator
            .get_relevant_visual_chunks(
                sample,
                processor,
            )
        )

        # 和 SigLIP 完全同一规则
        if not relevant:
            missing_relevance += 1
            total += 1
            continue

        retrieved = retrievals[
            qid
        ]

        hits += evaluator.recall_at_k(
            retrieved,
            relevant,
            TOP_K,
        )

        total += 1

        if total % 100 == 0:
            print(
                f"evaluated={total}, "
                f"hits={hits}, "
                f"missing_relevance="
                f"{missing_relevance}"
            )

    metrics = {
        f"recall@{TOP_K}": (
            hits / total
            if total
            else 0
        ),
        "hits": hits,
        "evaluated_questions": total,
        "target_questions": len(eval_ids),
        "missing_relevance": missing_relevance,
        "total_questions": len(samples),
    }

    save_json(
        RECALL_PATH,
        metrics,
    )

    print(
        "\n===== ColPali Recall ====="
    )

    print(
        json.dumps(
            metrics,
            indent=2,
        )
    )

def generate_predictions():
    processor = get_processor()

    samples = (
        processor.get_samples()
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

    def process(sample):
        qid = str(
            sample["questionId"]
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

            if i % 5 == 0:
                save_json(
                    PREDICTION_PATH,
                    predictions,
                )

    save_json(
        PREDICTION_PATH,
        predictions,
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


def run_all():
    print(
        "\n=== 1. ColPali Index ==="
    )

    build_index()

    print(
        "\n=== 2. ColPali Retrieval ==="
    )

    retrieve_all()

    print(
        "\n=== 3. Recall@5 ==="
    )

    evaluate_recall()

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

    args = parser.parse_args()

    if args.mode == "all":
        run_all()

    elif args.mode == "index":
        build_index()

    elif args.mode == "retrieve":
        retrieve_all()

    elif args.mode == "recall":
        evaluate_recall()

    elif args.mode == "generate":
        generate_predictions()


if __name__ == "__main__":
    main()