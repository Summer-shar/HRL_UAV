import argparse
import csv
import json
import os


DEFAULT_IGNORED_FIELDS = {
    "micro_decision_seconds",
    "micro_decision_mean_ms",
    "micro_decision_p95_ms",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare two evaluation CSVs under a fixed protocol."
    )
    parser.add_argument("--reference", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def read_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return {int(row["episode"]): row for row in csv.DictReader(handle)}


def compare_rows(reference_rows, candidate_rows, ignored_fields=None):
    ignored_fields = set(ignored_fields or ())
    comparisons = []
    all_match = True
    if set(reference_rows) != set(candidate_rows):
        all_match = False
    for episode in sorted(set(reference_rows) & set(candidate_rows)):
        reference = reference_rows[episode]
        candidate = candidate_rows[episode]
        mismatches = []
        for field in reference:
            if field in ignored_fields:
                continue
            if reference[field] != candidate.get(field):
                mismatches.append(
                    {
                        "field": field,
                        "reference": reference[field],
                        "candidate": candidate.get(field),
                    }
                )
        if mismatches:
            all_match = False
        comparisons.append(
            {
                "episode": episode,
                "scientific_fields_match": not mismatches,
                "mismatches": mismatches,
            }
        )
    return {
        "all_scientific_fields_match": all_match,
        "reference_episodes": sorted(reference_rows),
        "candidate_episodes": sorted(candidate_rows),
        "ignored_runtime_fields": sorted(ignored_fields),
        "by_episode": comparisons,
    }


def main():
    args = parse_args()
    result = compare_rows(
        read_rows(args.reference),
        read_rows(args.candidate),
        ignored_fields=DEFAULT_IGNORED_FIELDS,
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_scientific_fields_match"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
