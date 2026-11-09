"""Capstone Q3: alignment metrics between human labels and the LLM judge.

    uv run python evals/metrics.py evals/results_judged_v1.json

Positive class = "bad". Prints the confusion matrix, accuracy, precision and
recall, followed by every disagreement with the judge's reasoning (the raw
material for improving the judge prompt).
"""

import csv
import json
import sys
from pathlib import Path

EVALS = Path(__file__).parent


def main() -> None:
    judged_file = Path(sys.argv[1]) if len(sys.argv) > 1 else EVALS / 'results_judged_v1.json'

    with open(EVALS / 'labels.csv') as f:
        human = {int(r['index']): r for r in csv.DictReader(f)}

    with open(judged_file) as f:
        results = json.load(f)

    tp = fp = fn = tn = 0
    disagreements = []
    evaluated = 0

    for row in results:
        i = row['index']
        if i not in human or not human[i]['label']:
            continue  # only compare items I actually labeled
        evaluated += 1
        h = human[i]['label']
        j = row['judge_label']
        if h == 'bad' and j == 'bad':
            tp += 1
        elif h == 'good' and j == 'bad':
            fp += 1
            disagreements.append((row, h, j))
        elif h == 'bad' and j == 'good':
            fn += 1
            disagreements.append((row, h, j))
        else:
            tn += 1

    print(f"judge: {judged_file.name}   human labels: labels.csv ({evaluated} items)")
    print(f"TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"accuracy  = (TP+TN)/total = ({tp}+{tn})/{evaluated} = {(tp + tn) / evaluated:.4f}")
    print(f"precision = TP/(TP+FP) = {tp}/{tp + fp} = {tp / (tp + fp):.4f}" if tp + fp else "precision = n/a")
    print(f"recall    = TP/(TP+FN) = {tp}/{tp + fn} = {tp / (tp + fn):.4f}" if tp + fn else "recall = n/a")
    print(f"disagreements = {len(disagreements)}")

    for row, h, j in disagreements:
        print(f"\n#{row['index']} [{row['type']}] {row['question']}")
        print(f"  human={h} judge={j}")
        print(f"  judge reasoning: {row.get('judge_reasoning', '(none)')}")


if __name__ == '__main__':
    main()
