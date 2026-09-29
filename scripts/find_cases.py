import json
import re
from pathlib import Path
from contextlib import redirect_stdout
import io

ANN = "data/annotations/infographicsVQA_val_v1.0_withQT.json"

FILES = {
    "text": {
        "recall": "results/text_document_recall_details.json",
        "pred": "results/text_predictions.json",
    },
    "siglip": {
        "recall": "results/siglip_document_recall_details.json",
        "pred": "results/siglip_predictions.json",
    },
    "colpali": {
        "recall": "results/colpali_document_recall_details.json",
        "pred": "results/colpali_predictions.json",
    },
}


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def normalize_tokens(s):
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = s.replace(" and ", " ")
    return set(s.split())


def is_correct(pred, answers):
    p_norm = norm(pred)

    for a in answers:
        # 1. 原来的严格规范化匹配
        if p_norm == norm(a):
            return True

        # 2. 对多项答案允许顺序不同
        if normalize_tokens(pred) == normalize_tokens(a):
            return True

    return False


samples = load(ANN)["data"]

data = {}
for name, paths in FILES.items():
    data[name] = {
        "recall": load(paths["recall"]),
        "pred": load(paths["pred"]),
    }


all_good = []
colpali_improvement = []
retrieval_failure = []
generation_failure = []
faithfulness_candidates = []

uncertain_words = [
    "cannot answer",
    "not enough",
    "insufficient",
    "not mentioned",
    "not found",
    "unable",
]


for sample in samples:
    qid = str(sample["questionId"])

    if any(
        qid not in data[m]["recall"] or qid not in data[m]["pred"]
        for m in data
    ):
        continue

    item = {
        "qid": qid,
        "question": sample["question"],
        "gt": sample["answers"],
    }

    for method in ["text", "siglip", "colpali"]:
        hit = data[method]["recall"][qid]["hit"]
        pred = data[method]["pred"][qid]["answer"]
        correct = is_correct(pred, sample["answers"])

        item[method] = {
            "hit": hit,
            "prediction": pred,
            "correct": correct,
        }

    # 1. 三种方法都成功
    if all(
        item[m]["hit"] and item[m]["correct"]
        for m in ["text", "siglip", "colpali"]
    ):
        all_good.append(item)

    # 2. ColPali 明显优于 baseline
    if (
        item["colpali"]["hit"]
        and item["colpali"]["correct"]
        and (
            not item["siglip"]["correct"]
            or not item["siglip"]["hit"]
            or not item["text"]["correct"]
        )
    ):
        colpali_improvement.append(item)

    # 3. ColPali retrieval failure
    if (
        not item["colpali"]["hit"]
        and not item["colpali"]["correct"]
    ):
        retrieval_failure.append(item)

    # 4. ColPali retrieval hit 但回答错误
    if (
        item["colpali"]["hit"]
        and not item["colpali"]["correct"]
    ):
        generation_failure.append(item)

    # 5. Faithfulness / overconfidence 候选
    pred = item["colpali"]["prediction"].lower()

    if (
        not item["colpali"]["hit"]
        and not item["colpali"]["correct"]
        and not any(x in pred for x in uncertain_words)
    ):
        faithfulness_candidates.append(item)


def show(title, items, n=10):
    print(f"\n=== {title} ===")

    for x in items[:n]:
        print("\nQID:", x["qid"])
        print("Question:", x["question"])
        print("GT:", x["gt"])

        for method in ["text", "siglip", "colpali"]:
            m = x[method]

            print(
                f"{method:8s} | "
                f"hit={m['hit']} | "
                f"correct={m['correct']} | "
                f"pred={m['prediction']}"
            )


output_buffer = io.StringIO()

with redirect_stdout(output_buffer):
    show(
        "ALL GOOD",
        all_good
    )

    show(
        "COLPALI IMPROVEMENT",
        colpali_improvement
    )

    show(
        "RETRIEVAL FAILURE",
        retrieval_failure
    )

    show(
        "GENERATION FAILURE CANDIDATES",
        generation_failure
    )

    show(
        "FAITHFULNESS / OVERCONFIDENCE CANDIDATES",
        faithfulness_candidates
    )


output = output_buffer.getvalue()

# 仍然打印到终端
print(output, end="")

# 同时保存
output_path = Path(
    "results/case_study_candidates.txt"
)

output_path.parent.mkdir(
    parents=True,
    exist_ok=True
)

output_path.write_text(
    output,
    encoding="utf-8"
)

print(
    f"\nSaved case study candidates to: "
    f"{output_path}"
)