"""Validate fixed-workload timing evidence and write JSON plus Markdown.

Only metadata.json, times.json and completion.json are read. No torch import,
trace parsing, measurement, simulation, training, or candidate selection occurs.
--output accepts a .json/.md filename or a filename stem; both formats are saved.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics


SCHEMA = "equivalent_performance_comparison_1"
MEMORY_FIELDS = ("rss_peak", "cuda_peak_allocated", "cuda_peak_reserved")
TAIL_NOTE = ("Raw repetitions and medians describe these fixed workloads. "
             "The small repetition count does not establish tail latency; no P95 claim is made.")
CANDIDATE_NOTE = ("Diagnostic candidate only. Numerical equivalence acceptance is not supplied "
                  "to this tool. A measured speed ratio is not approval to change the default.")


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def read_json(path):
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    canonical(value)  # Reject NaN/Infinity even if the JSON reader accepts them.
    if not isinstance(value, dict):
        raise ValueError(str(path) + " must contain a JSON object")
    return value


def first_difference(left, right, path="workload"):
    """A short, concrete mismatch location without printing enormous reports."""
    if type(left) is not type(right):
        return path + ": JSON types differ"
    if isinstance(left, dict):
        if left.keys() != right.keys():
            return path + ": keys differ"
        for key in sorted(left):
            difference = first_difference(left[key], right[key], path + "." + key)
            if difference:
                return difference
    elif isinstance(left, list):
        if len(left) != len(right):
            return path + ": lengths differ"
        for index, (a, b) in enumerate(zip(left, right)):
            difference = first_difference(a, b, path + "[" + str(index) + "]")
            if difference:
                return difference
    elif left != right:
        return path + ": values differ"
    return None


def finite_number(value, name, *, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(name + " must be a finite number")
    if (positive and value <= 0) or (not positive and value < 0):
        raise ValueError(name + (" must be positive" if positive else " must be nonnegative"))
    return value


def sha_string(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(name + " must be a lowercase SHA256 digest")
    return value


def summarize_row(name, row):
    if not isinstance(row, dict) or row.get("name") != name:
        raise ValueError(name + ": row.name does not match its times.json key")
    if not isinstance(row.get("workload"), dict):
        raise ValueError(name + ": missing workload object")
    if not isinstance(row.get("counts_after"), dict) or not row["counts_after"]:
        raise ValueError(name + ": missing update counters")
    samples = row.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError(name + ": missing raw timing repetitions")
    if type(row.get("repeats")) is not int or row["repeats"] != len(samples):
        raise ValueError(name + ": repeats does not match the raw sample count")
    seconds = []
    memory = {key: [] for key in MEMORY_FIELDS}
    missing_memory = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError(name + ": timing sample must be an object")
        seconds.append(finite_number(sample.get("seconds"), name + ".seconds", positive=True))
        reading = sample.get("memory", {})
        if not isinstance(reading, dict):
            raise ValueError(name + ": memory reading must be an object")
        for key in MEMORY_FIELDS:
            if key not in reading or reading[key] is None:
                missing_memory.append(dict(sample=index, field=key))
            else:
                memory[key].append(finite_number(reading[key], name + ".memory." + key))
    measured_median = statistics.median(seconds)
    reported_median = finite_number(row.get("median_s"), name + ".median_s", positive=True)
    if not math.isclose(measured_median, reported_median, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError(name + ": median_s disagrees with the raw timing samples")
    return dict(raw_seconds=seconds, median_s=reported_median,
                repeats=len(seconds), warmup=row.get("warmup"),
                memory_max_bytes={key: max(values) if len(values) == len(samples) else None
                                  for key, values in memory.items()},
                missing_memory=missing_memory, counts_after_recorded=row.get("counts_after"),
                tail_estimate_reliable=False)


def load_run(path, role):
    path = path.resolve()
    metadata = read_json(path / "metadata.json")
    times = read_json(path / "times.json")
    completion = read_json(path / "completion.json")
    if completion.get("status") != "complete":
        raise ValueError(role + ": completion.status is not complete")
    completed_cases = completion.get("cases")
    if (not isinstance(completed_cases, list) or
            any(not isinstance(name, str) for name in completed_cases) or
            len(completed_cases) != len(set(completed_cases)) or set(completed_cases) != set(times)):
        raise ValueError(role + ": completion case inventory differs from times.json")
    if not times:
        raise ValueError(role + ": completed run has no measured cases")
    for key in ("artifact_sha256", "initial_state_sha256"):
        sha_string(metadata.get(key), role + ".metadata." + key)
    if completion.get("fixture_sha256") != metadata["artifact_sha256"]:
        raise ValueError(role + ": completion fixture digest differs from metadata")
    if not isinstance(metadata.get("config"), dict):
        raise ValueError(role + ": missing exact configuration")
    source = metadata.get("source")
    if not isinstance(source, dict) or not source:
        raise ValueError(role + ": missing production source manifest")
    for name, value in source.items():
        sha_string(value, role + ".source." + name)
    rows = {name: summarize_row(name, row) for name, row in times.items()}
    return dict(role=role, directory=str(path), metadata=metadata, times=times,
                completion=completion, rows=rows)


def run_identity(run):
    metadata = run["metadata"]
    return dict(role=run["role"], label=metadata.get("label"), directory=run["directory"],
        data=dict(artifact_sha256=metadata["artifact_sha256"],
                  initial_state_sha256=metadata["initial_state_sha256"],
                  config_sha256=digest(metadata["config"])),
        research_code=dict(source_manifest_sha256=digest(metadata["source"]),
                           source_files=metadata["source"]),
        measurement_code=dict(script_sha256=metadata.get("measurement_script_sha256")),
        runtime=metadata.get("runtime"), cpu_threads=metadata.get("cpu_threads"),
        deterministic=metadata.get("deterministic"), microbatches=metadata.get("microbatches"),
        initialization=metadata.get("initial_state_preparation"),
        initial_optimizer_slots=metadata.get("initial_optimizer_slots"),
        initial_counts=metadata.get("initial_counts"),
        physical_steps_this_run=metadata.get("physical_steps_this_run"),
        candidate_diagnostic_only=run["role"].startswith("candidate"))


def validate_contracts(runs):
    reference = runs[0]
    errors = []
    for run in runs[1:]:
        for key in ("artifact_sha256", "initial_state_sha256", "config", "runtime",
                    "cpu_threads", "deterministic", "measurement_script_sha256",
                    "initial_counts", "initial_optimizer_slots"):
            difference = first_difference(reference["metadata"][key], run["metadata"][key], key)
            if difference:
                errors.append(dict(left=reference["role"], right=run["role"], reason=difference))
        for key in ("target", "actor", "ppo"):
            difference = first_difference(reference["metadata"]["microbatches"][key],
                                          run["metadata"]["microbatches"][key], "microbatches." + key)
            if difference:
                errors.append(dict(left=reference["role"], right=run["role"], reason=difference))
    # Validate every shared case across all runs, including a case absent from
    # baseline but present in both current and a candidate.
    owners = {}
    for run in runs:
        for name, row in run["times"].items():
            if name in owners:
                other, workload, counts = owners[name]
                difference = first_difference(workload, row["workload"])
                if difference:
                    errors.append(dict(case=name, left=other, right=run["role"], reason=difference))
                difference = first_difference(counts, row["counts_after"], "counts_after")
                if difference:
                    errors.append(dict(case=name, left=other, right=run["role"], reason=difference))
            else:
                owners[name] = (run["role"], row["workload"], row["counts_after"])
    return errors


def compare(before, after):
    shared = sorted(set(before["rows"]) & set(after["rows"]))
    missing = []
    for name in sorted(set(before["rows"]) ^ set(after["rows"])):
        missing.append(dict(case=name,
            missing_from=before["role"] if name not in before["rows"] else after["role"],
            available_from=after["role"] if name not in before["rows"] else before["role"],
            available_measurement=(after["rows"].get(name) or before["rows"][name]),
            speedup=None, status="missing"))
    measurements = []
    for name in shared:
        left, right = before["rows"][name], after["rows"][name]
        measurements.append(dict(case=name, status="comparable", workload=before["times"][name]["workload"],
            before=left, after=right, speedup=left["median_s"] / right["median_s"],
            speedup_definition="before median / after median; greater than 1 means faster after",
            counts_after_equal=(left["counts_after_recorded"] == right["counts_after_recorded"])))
    notes = []
    for key in ("runtime", "cpu_threads", "deterministic", "measurement_script_sha256"):
        if before["metadata"].get(key) != after["metadata"].get(key):
            notes.append(key + " differs; inspect the separately recorded identities before attributing a speed ratio")
    candidate = after["role"].startswith("candidate")
    return dict(before=before["role"], after=after["role"], shared_cases=measurements,
                missing_cases=missing, notes=notes,
                candidate_diagnostic_only=candidate,
                numerical_acceptance="not_assessed_by_this_tool",
                default_selection=None, candidate_note=CANDIDATE_NOTE if candidate else None)


def escape(value):
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def seconds_cell(row):
    raw = ", ".join(format(value, ".9g") for value in row["raw_seconds"])
    return format(row["median_s"], ".9g") + " [" + raw + "]"


def memory_cell(before, after, key):
    def one(row):
        value = row["memory_max_bytes"][key]
        return "missing" if value is None else format(value / 2**20, ".2f")
    return one(before) + " → " + one(after)


def markdown(summary):
    lines = ["# Fixed-workload performance comparison", "", TAIL_NOTE, "",
        "RSS peaks are sampled process memory; CUDA allocated/reserved peaks include the resident model. "
        "These are observed maxima across repetitions, not memory deltas.", "",
        "Data identity and research-code identity are recorded separately. Production source changes are allowed; "
        "fixture, initial state, configuration, shared workloads and update counters must match exactly.", ""]
    if summary["status"] != "complete":
        lines += ["Validation failed. No speed comparison has been produced.", ""]
        for error in summary["errors"]:
            lines.append("- " + escape(json.dumps(error, ensure_ascii=False)))
        return "\n".join(lines) + "\n"
    data = summary["data_identity"]
    lines += ["- Fixture SHA256: `" + data["artifact_sha256"] + "`",
              "- Initial state SHA256: `" + data["initial_state_sha256"] + "`",
              "- Configuration SHA256: `" + data["config_sha256"] + "`", "",
              "| Run | Research source manifest SHA256 | Critic microbatch |",
              "| --- | --- | ---: |"]
    for run in summary["runs"]:
        lines.append("| " + escape(run["role"]) + " | " + run["research_code"]["source_manifest_sha256"] +
                     " | " + escape((run.get("microbatches") or {}).get("critic", "missing")) + " |")
    for comparison in summary["comparisons"]:
        lines += ["", "## " + escape(comparison["before"]) + " → " + escape(comparison["after"]), ""]
        if comparison["candidate_diagnostic_only"]:
            lines += [CANDIDATE_NOTE, ""]
        for note in comparison["notes"]:
            lines += ["- " + escape(note)]
        lines += ["", "Times are seconds: **median [raw repetitions]**. Speedup is before/after median. "
                  "Memory columns are before → after in MiB.", "",
            "| Case | Before seconds | After seconds | Speedup | Repeats | Peak RSS | Peak allocated | Peak reserved |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
        for case in comparison["shared_cases"]:
            left, right = case["before"], case["after"]
            cells = [case["case"], seconds_cell(left), seconds_cell(right),
                     format(case["speedup"], ".4f") + "×", str(left["repeats"]) + "/" + str(right["repeats"])]
            cells.extend(memory_cell(left, right, key) for key in MEMORY_FIELDS)
            lines.append("| " + " | ".join(escape(cell) for cell in cells) + " |")
        if not comparison["shared_cases"]:
            lines += ["", "No shared measured cases; no speedup is available."]
        if comparison["missing_cases"]:
            lines += ["", "Missing measurements (no values substituted):", ""]
            for case in comparison["missing_cases"]:
                lines.append("- `" + escape(case["case"]) + "`: missing from " + escape(case["missing_from"]) + ".")
    lines += ["", "Update counters must match exactly and are retained as records; different PPO stopping or update counts reject comparison. "
              "This report does not establish numerical equivalence and never changes or recommends a default automatically.", ""]
    return "\n".join(lines)


def write_outputs(path, summary, protected):
    path = path.resolve()
    if path.suffix.lower() == ".md":
        md_path, json_path = path, path.with_suffix(".json")
    else:
        json_path = path if path.suffix.lower() == ".json" else Path(str(path) + ".json")
        md_path = json_path.with_suffix(".md")
    for target in (json_path, md_path):
        if target in protected or target.is_dir():
            raise ValueError("Output would overwrite an input artifact or directory: " + str(target))
    texts = [(json_path, json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n"),
             (md_path, markdown(summary))]
    for target, text in texts:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + "." + str(os.getpid()) + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, target)
    return json_path, md_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = [("baseline", args.baseline), ("current", args.current)]
    paths += [("candidate_" + str(index), path) for index, path in enumerate(args.candidate, 1)]
    protected = {path.resolve() / name for _, path in paths
                 for name in ("metadata.json", "times.json", "completion.json")}
    runs, errors = [], []
    for role, path in paths:
        try:
            runs.append(load_run(path, role))
        except (OSError, ValueError, TypeError, KeyError) as error:
            errors.append(dict(run=role, reason=str(error)))
    if not errors:
        errors.extend(validate_contracts(runs))
    summary = dict(schema=SCHEMA, generated_utc=datetime.now(timezone.utc).isoformat(),
        status="rejected" if errors else "complete", errors=errors,
        tail_estimate_reliable=False, tail_note=TAIL_NOTE,
        numerical_acceptance="not_assessed_by_this_tool", default_selection=None,
        candidate_note=CANDIDATE_NOTE, runs=[run_identity(run) for run in runs], comparisons=[])
    if not errors:
        reference = runs[0]["metadata"]
        summary["data_identity"] = dict(artifact_sha256=reference["artifact_sha256"],
            initial_state_sha256=reference["initial_state_sha256"],
            config_sha256=digest(reference["config"]), config=reference["config"])
        summary["case_inventory"] = {name: {run["role"]: name in run["rows"] for run in runs}
                                     for name in sorted(set().union(*(set(run["rows"]) for run in runs)))}
        summary["comparisons"].append(compare(runs[0], runs[1]))
        for candidate in runs[2:]:
            summary["comparisons"].append(compare(runs[0], candidate))
            summary["comparisons"].append(compare(runs[1], candidate))
    json_path, md_path = write_outputs(args.output, summary, protected)
    print(json.dumps(dict(status=summary["status"], json=str(json_path), markdown=str(md_path))))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
