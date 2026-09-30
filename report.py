"""Build a CSV summary and self-contained HTML charts from JSONL benchmark runs."""

import argparse
import csv
import html
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def percentile(values, percent):
    """Nearest-rank percentile; small samples remain visibly small in reports."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percent / 100 * len(ordered)) - 1)]


def summarize(records):
    groups = defaultdict(list)
    for record in records:
        label = record.get("hardware", {}).get("device_label", "unspecified")
        key = (record.get("run_id", "legacy"), record.get("model", "unknown"),
               label, record.get("phase", "warm"))
        groups[key].append(record)
    rows = []
    for (run_id, model, device, phase), items in sorted(groups.items()):
        good = [r for r in items if "error" not in r]
        def nums(name):
            return [r[name] for r in good if isinstance(r.get(name), (int, float))]
        scored = [r for r in good if r.get("quality", {}).get("status") in ("pass", "fail")]
        by_prompt = defaultdict(list)
        for record in scored:
            by_prompt[record.get("prompt_id", "unknown")].append(record["quality"]["status"])
        rows.append({
            "run_id": run_id, "model": model, "device": device, "phase": phase,
            "digest12": (items[0].get("model_info", {}).get("digest") or "")[:12],
            "temperature": items[0].get("temperature"),
            "num_predict": items[0].get("num_predict"),
            "prompt_set_sha256": items[0].get("prompt_set_sha256"),
            "samples": len(good), "errors": len(items) - len(good),
            "scored": len(scored),
            "distinct_scored_prompts": len(by_prompt),
            "distinct_prompts_all_pass": sum(all(status == "pass" for status in statuses)
                                             for statuses in by_prompt.values()),
            "pass_rate_pct": round(100 * sum(r["quality"]["status"] == "pass" for r in scored) / len(scored), 1) if scored else None,
            "first_text_p50_ms": round(1000 * statistics.median(nums("first_text_s")), 1) if nums("first_text_s") else None,
            "first_text_p95_ms": round(1000 * percentile(nums("first_text_s"), 95), 1) if nums("first_text_s") else None,
            "wall_p50_ms": round(1000 * statistics.median(nums("wall_latency_s")), 1) if nums("wall_latency_s") else None,
            "wall_p95_ms": round(1000 * percentile(nums("wall_latency_s"), 95), 1) if nums("wall_latency_s") else None,
            "generation_tok_s_median": round(statistics.median(nums("tokens_per_s")), 1) if nums("tokens_per_s") else None,
            "load_p50_ms": round(1000 * statistics.median(nums("load_s")), 1) if nums("load_s") else None,
        })
    return rows


def chart(rows, field, title, suffix=""):
    visible = [(f"{r['model']} / {r['device']} / {r['run_id'][:8]} / {r['phase']}", r[field])
               for r in rows if isinstance(r[field], (int, float))]
    if not visible:
        return "<p>No values available.</p>"
    max_value = max(value for _, value in visible) or 1
    width = 700
    height = 48 + len(visible) * 42
    parts = [f'<svg role="img" aria-label="{html.escape(title)}" viewBox="0 0 {width} {height}">']
    for index, (label, value) in enumerate(visible):
        y = 26 + index * 42
        parts.append(f'<text x="4" y="{y}" font-size="12">{html.escape(label)}</text>')
        parts.append(f'<rect x="4" y="{y+5}" width="{max(2, round(560*value/max_value))}" height="17" rx="3" fill="#1f75ae"/>')
        parts.append(f'<text x="580" y="{y+19}" font-size="12">{value}{html.escape(suffix)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def write_report(input_path, output_dir):
    records = [json.loads(line) for line in Path(input_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise ValueError("No benchmark records found")
    rows = summarize(records)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / "summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    headings = list(rows[0])
    table = "<table><thead><tr>" + "".join(f"<th>{html.escape(h)}</th>" for h in headings) + "</tr></thead><tbody>"
    for row in rows:
        table += "<tr>" + "".join(f"<td>{html.escape(str(row[h])) if row[h] is not None else '—'}</td>" for h in headings) + "</tr>"
    table += "</tbody></table>"
    categories = defaultdict(lambda: {"pass": 0, "fail": 0, "unscored": 0, "error": 0})
    for record in records:
        if record.get("phase") != "warm":
            continue
        key = (record.get("run_id", "legacy")[:8],
               record.get("model", "unknown"),
               record.get("hardware", {}).get("device_label", "unspecified"),
               record.get("category", "unknown"))
        status = "error" if "error" in record else record.get("quality", {}).get("status", "unscored")
        categories[key][status] += 1
    category_table = "<table><thead><tr><th>Run / model / device / category</th><th>Pass</th><th>Fail</th><th>Unscored</th><th>Error</th></tr></thead><tbody>"
    for key, counts in sorted(categories.items()):
        category_table += f"<tr><td>{html.escape(' / '.join(key))}</td>" + "".join(f"<td>{counts[k]}</td>" for k in ("pass", "fail", "unscored", "error")) + "</tr>"
    category_table += "</tbody></table>"
    charts = [("First visible text p50 (ms)", "first_text_p50_ms", " ms"),
              ("Wall latency p50 (ms)", "wall_p50_ms", " ms"),
              ("Generation throughput median (tokens/s)", "generation_tok_s_median", " tok/s"),
              ("Strict pass rate", "pass_rate_pct", "%")]
    body = "".join(f"<section><h2>{html.escape(title)}</h2>{chart(rows, field, title, suffix)}</section>"
                   for title, field, suffix in charts)
    html_path = destination / "report.html"
    html_path.write_text(f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Edge LLM benchmark report</title><style>body{{font:15px system-ui,sans-serif;max-width:1100px;margin:40px auto;padding:0 16px;color:#172334}}
h1,h2{{color:#12395a}}section{{margin:28px 0}}svg{{width:100%;max-width:700px}}table{{border-collapse:collapse;display:block;overflow-x:auto}}
th,td{{border:1px solid #ccd7df;padding:6px 10px;text-align:right;white-space:nowrap}}th:first-child,td:first-child{{text-align:left}}
th{{background:#eaf2f7}}.note{{background:#fff5d8;padding:12px;border-left:4px solid #c78a1a}}</style>
<h1>Edge LLM benchmark report</h1><p>Input: {html.escape(str(input_path))} · {len(records)} recorded requests</p>
<p class="note">First visible text is a client-side TTFT proxy, not internal first-token timing. p95 uses nearest rank. 
The median generation rate is based on Ollama's output token count and eval duration; tokenizers differ across models. 
Strict pass rate covers only automatically scored prompts. Inspect sample counts and errors before comparing models.</p>
{table}<section><h2>Quality by prompt category</h2>{category_table}</section>{body}</html>""", encoding="utf-8")
    return csv_path, html_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="JSONL generated by benchmark.py")
    parser.add_argument("--out-dir", default="results/report")
    args = parser.parse_args()
    for path in write_report(args.input, args.out_dir):
        print(path)


if __name__ == "__main__":
    main()
