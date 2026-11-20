"""Join completed evaluation artifacts to the experiment matrix without fabricating scores."""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--matrix", type=Path, default=Path("configs/experiment-matrix.csv"))
    parser.add_argument("--output", type=Path, default=Path("reports/evaluation.json"))
    args = parser.parse_args()
    with args.matrix.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        filename = "metrics.json" if row["task"] == "detect" else "test.json"
        path = args.runs / row["id"] / filename
        if path.is_file():
            row["evaluation"] = json.loads(path.read_text())
            row["status"] = "evaluated"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
