import json
from pathlib import Path

PRED_PATH = Path("results/text_predictions.json")
TEMPLATE_PATH = Path("eval/submissions/empty_results.json")
OUTPUT_PATH = Path("eval/submissions/text_rag.json")


def main():
    with open(PRED_PATH, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
        template = json.load(f)

    submission = []

    missing = 0

    for item in template:
        qid = str(item["questionId"])

        if qid in predictions:
            answer = predictions[qid]["answer"]
        else:
            answer = ""
            missing += 1

        submission.append({
            "questionId": item["questionId"],
            "answer": answer,
        })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(
            submission,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Submission size: {len(submission)}")
    print(f"Missing predictions: {missing}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()