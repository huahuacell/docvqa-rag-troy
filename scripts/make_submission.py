import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--predictions",
        required=True,
        help="Path to predictions JSON",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path to output submission JSON",
    )

    parser.add_argument(
        "--template",
        default="eval/submissions/empty_results.json",
        help="Path to official empty submission template",
    )

    args = parser.parse_args()

    pred_path = Path(args.predictions)
    template_path = Path(args.template)
    output_path = Path(args.output)

    with open(pred_path, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    with open(template_path, "r", encoding="utf-8") as f:
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

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            submission,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Submission size: {len(submission)}")
    print(f"Missing predictions: {missing}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()