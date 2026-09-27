"""Summarize Chrome trace annotations without treating them as wall-clock shares.

Run after all wall-clock measurements finish. This offline command reads the
trace into memory using only the Python standard library; it imports no model,
CUDA library or profiler. --output is a filename prefix or a .json/.md path.
"""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path


SCHEMA = "chrome_trace_inclusive_cpu_summary_1"
CUDA_CATEGORIES = {
    "kernel": {"kernel", "gpu_kernel", "cuda_kernel"},
    "memcpy": {"gpu_memcpy", "cuda_memcpy"},
    "memset": {"gpu_memset", "cuda_memset"},
}
TRANSFER_ATEN_NAMES = {
    "aten::to", "aten::_to_copy", "aten::copy_", "aten::copy",
    "aten::_copy_from", "aten::_copy_from_and_resize",
}
TRANSFER_RUNTIME_PREFIXES = ("cudaMemcpy", "cuMemcpy", "cudaMemset", "cuMemset")
NOTES = [
    "All reported CPU durations are inclusive nested spans. Rows overlap and must not be added together.",
    "These tables are not a wall-clock decomposition and contain no wall-clock percentage attribution.",
    "Transfer-related CPU calls measure host dispatch/conversion/copy work, not actual CUDA transfer duration; aten::to may only change dtype or return its input.",
    "CUDA counts come only from actual kernel/gpu_memcpy/gpu_memset categories, never cudaLaunchKernel or other CPU runtime API calls.",
    "Chrome trace ts/dur fields are interpreted as microseconds; displayTimeUnit does not change their storage unit.",
    "Only completed X or matched B/E spans contribute duration. Instant annotation events are counted with zero duration; incomplete/unsupported annotation events are reported separately.",
    "This offline parser loads the trace into host memory. Run it after performance measurements, not alongside them.",
]


def categories(event):
    return {value.strip().lower() for value in str(event.get("cat", "")).split(",") if value.strip()}


def cuda_kind(event):
    found = categories(event)
    for kind, names in CUDA_CATEGORIES.items():
        if found & names:
            return kind
    return None


def seconds(duration):
    if not isinstance(duration, (int, float)) or isinstance(duration, bool):
        return None
    value = float(duration)
    return value / 1_000_000 if math.isfinite(value) and value >= 0 else None


def summarize(document, trace):
    events = document.get("traceEvents") if isinstance(document, dict) else document
    if not isinstance(events, list):
        raise ValueError("Expected a Chrome trace object with traceEvents, or an event array")
    annotations = defaultdict(lambda: dict(calls=0, inclusive_cpu_seconds=0., instant_events=0))
    transfers = defaultdict(lambda: dict(calls=0, inclusive_cpu_seconds=0., instant_events=0))
    gpu = dict(kernel=0, memcpy=0, memset=0)
    stacks = defaultdict(list)
    ignored = defaultdict(int)
    invalid_durations = 0

    def record(event, duration, instant=False):
        nonlocal invalid_durations
        elapsed = seconds(duration)
        if elapsed is None:
            invalid_durations += 1
            return
        kind = cuda_kind(event)
        if kind is not None:
            gpu[kind] += 1
            return
        cats = categories(event)
        # GPU annotations are separate from the CPU record_function spans.
        if "gpu_user_annotation" in cats:
            return
        name = str(event.get("name", "<unnamed>"))
        tables = []
        if "user_annotation" in cats:
            tables.append(annotations)
        # Match aten overloads without accidentally including aten::topk or
        # aten::copysign as transfer operations.
        if name.split(".", 1)[0] in TRANSFER_ATEN_NAMES or name.startswith(TRANSFER_RUNTIME_PREFIXES):
            tables.append(transfers)
        for table in tables:
            row = table[name]
            row["calls"] += 1
            row["inclusive_cpu_seconds"] += elapsed
            row["instant_events"] += int(instant)

    for event in events:
        if not isinstance(event, dict):
            ignored["non_object_events"] += 1
            continue
        phase = event.get("ph")
        if phase == "X":
            record(event, event.get("dur"))
        elif phase == "B":
            stacks[(event.get("pid"), event.get("tid"))].append(event)
        elif phase == "E":
            stack = stacks[(event.get("pid"), event.get("tid"))]
            if not stack:
                ignored["unmatched_end_events"] += 1
                continue
            start = stack.pop()
            if isinstance(start.get("ts"), (int, float)) and isinstance(event.get("ts"), (int, float)):
                record(start, event["ts"] - start["ts"])
            else:
                invalid_durations += 1
        elif "user_annotation" in categories(event):
            if phase in ("i", "I"):
                record(event, 0, instant=True)
            else:
                ignored["user_annotation_phase_" + str(phase)] += 1
    for stack in stacks.values():
        for event in stack:
            ignored["unmatched_begin_events"] += 1
            if "user_annotation" in categories(event):
                ignored["incomplete_user_annotations"] += 1

    def rows(table):
        return sorted((dict(name=name, **value) for name, value in table.items()),
                      key=lambda value: (-value["inclusive_cpu_seconds"], value["name"]))

    activity_available = bool(gpu["kernel"] + gpu["memcpy"])
    status = (
        "Actual CUDA kernel/memcpy activity events are present. Counts alone do not attribute wall-clock time."
        if activity_available else
        "No actual CUDA kernel/memcpy events were captured. CUPTI activity data is unavailable in this trace; do not attribute time to CUDA kernels or transfers."
    )
    return dict(schema=SCHEMA, trace_path=str(trace.resolve()), trace_bytes=trace.stat().st_size,
                events_seen=len(events), user_annotations=rows(annotations),
                cpu_transfer_related_calls=rows(transfers),
                cuda_activity=dict(kernel_events=gpu["kernel"], memcpy_events=gpu["memcpy"],
                                   memset_events=gpu["memset"], activity_available=activity_available,
                                   interpretation=status),
                diagnostics=dict(invalid_durations=invalid_durations, ignored_events=dict(ignored)),
                notes=NOTES)


def markdown_escape(value):
    return str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")


def markdown(summary):
    lines = ["# Chrome trace CPU summary", "", "Trace: " + markdown_escape(summary["trace_path"]), "",
             "All durations below are **nested inclusive CPU spans**. They overlap and cannot be summed or interpreted as wall-clock shares.", "",
             "## CUDA activity", ""]
    activity = summary["cuda_activity"]
    lines += [f"Kernel events: {activity['kernel_events']}; memcpy events: {activity['memcpy_events']}; memset events: {activity['memset_events']}.",
              "", activity["interpretation"], ""]
    for title, key in (("User annotations", "user_annotations"),
                       ("Transfer-related CPU calls", "cpu_transfer_related_calls")):
        lines += ["## " + title, ""]
        if not summary[key]:
            lines += ["No matching completed spans or instant annotations.", ""]
            continue
        lines += ["| Name | Calls | Inclusive CPU seconds | Instant events |",
                  "| --- | ---: | ---: | ---: |"]
        for row in summary[key]:
            lines.append(f"| {markdown_escape(row['name'])} | {row['calls']} | {row['inclusive_cpu_seconds']:.9f} | {row['instant_events']} |")
        lines.append("")
    lines += ["## Interpretation and coverage", ""]
    lines.extend("- " + note for note in summary["notes"])
    lines += ["", f"Events read: {summary['events_seen']}; invalid durations: {summary['diagnostics']['invalid_durations']}.",
              "", "Ignored/incomplete events: `" + json.dumps(summary["diagnostics"]["ignored_events"], sort_keys=True) + "`.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="Output filename prefix, or a .json/.md path; both sibling files are written")
    args = parser.parse_args()
    prefix = args.output.with_suffix("") if args.output.suffix.lower() in (".json", ".md") else args.output
    json_path = Path(str(prefix) + ".json")
    md_path = Path(str(prefix) + ".md")
    if args.trace.resolve() in (json_path.resolve(), md_path.resolve()):
        parser.error("Output paths must not replace the input trace")
    with args.trace.open("r", encoding="utf-8-sig") as stream:
        summary = summarize(json.load(stream), args.trace)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    md_path.write_text(markdown(summary), encoding="utf-8")
    print(json.dumps(dict(json=str(json_path.resolve()), markdown=str(md_path.resolve()),
                          cuda_activity=summary["cuda_activity"]), ensure_ascii=False))


if __name__ == "__main__":
    main()
