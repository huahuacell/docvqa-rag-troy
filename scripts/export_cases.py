import json
import shutil
from pathlib import Path


CASE_IDS = [
    "98437",  # Good Case
    "91297",  # Retrieval Failure
    "91220",  # Faithfulness / Overconfidence
    "98322",  # ColPali Improvement
    "98370",  # Correct but potentially unfaithful
]


ANN_PATH = Path(
    "data/annotations/infographicsVQA_val_v1.0_withQT.json"
)

CHUNK_DIR = Path("data/chunks")

OUT_DIR = Path("results/case_studies")


FILES = {
    "text": {
        "details": Path(
            "results/text_document_recall_details.json"
        ),
        "pred": Path(
            "results/text_predictions.json"
        ),
    },

    "siglip": {
        "details": Path(
            "results/siglip_document_recall_details.json"
        ),
        "pred": Path(
            "results/siglip_predictions.json"
        ),
    },

    "colpali": {
        "details": Path(
            "results/colpali_document_recall_details.json"
        ),
        "pred": Path(
            "results/colpali_predictions.json"
        ),
    },
}


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


annotations = load_json(ANN_PATH)["data"]

ann_map = {
    str(x["questionId"]): x
    for x in annotations
}


data = {}

for method, paths in FILES.items():
    data[method] = {
        "details": load_json(paths["details"]),
        "pred": load_json(paths["pred"]),
    }


OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


for qid in CASE_IDS:

    print()
    print("=" * 60)
    print("CASE:", qid)
    print("=" * 60)

    case_dir = OUT_DIR / qid

    case_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    sample = ann_map[qid]

    summary = {
        "questionId": sample["questionId"],
        "question": sample["question"],
        "ground_truth": sample["answers"],
        "target_image": sample["image_local_name"],
        "methods": {},
    }

    print("Question:", sample["question"])
    print("GT:", sample["answers"])
    print("Target image:", sample["image_local_name"])

    for method in [
        "text",
        "siglip",
        "colpali"
    ]:

        details = data[method]["details"][qid]
        pred = data[method]["pred"][qid]

        print()
        print(
            f"[{method.upper()}] "
            f"hit={details['hit']} "
            f"pred={pred['answer']}"
        )

        method_info = {
            "hit": details["hit"],
            "prediction": pred["answer"],
            "retrieved": [],
        }

        for item in details["retrieved"]:

            retrieved_item = {
                "rank": item["rank"],
                "image_name": item["image_name"],
                "chunk_id": item.get("chunk_id"),
                "score": item.get("score"),
                "bbox": item.get("bbox"),
            }

            # Text RAG 可能还有 OCR text
            if "text" in item:
                retrieved_item["text"] = item["text"]

            method_info["retrieved"].append(
                retrieved_item
            )

            print(
                f"  rank={item['rank']} "
                f"image={item['image_name']} "
                f"chunk={item.get('chunk_id')} "
                f"score={item.get('score')}"
            )

            # Text RAG 不需要复制 visual chunk
            if method == "text":
                continue

            chunk_id = item.get("chunk_id")

            if not chunk_id:
                print(
                    "  [WARN] no chunk_id"
                )
                continue

            # 当前项目 chunk 命名：
            # data/chunks/37313_0001.jpg
            src = CHUNK_DIR / f"{chunk_id}.jpg"

            if not src.exists():
                print(
                    f"  [WARN] missing chunk: {src}"
                )
                continue

            dst = (
                case_dir /
                f"{method}_rank{item['rank']}_{chunk_id}.jpg"
            )

            shutil.copy2(
                src,
                dst
            )

        summary["methods"][method] = method_info

    # 保存 case summary
    summary_path = case_dir / "summary.json"

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2
        )

    print()
    print(
        "Saved:",
        summary_path
    )


print()
print("=" * 60)
print(
    "All case studies exported to:",
    OUT_DIR
)
print("=" * 60)