"""Repeatable, single-client Ollama text-generation benchmark."""

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

from evaluator import evaluate
from hardware import collect


def get_json(base_url, path, timeout=10):
    with urllib.request.urlopen(base_url + path, timeout=timeout) as response:
        return json.load(response)


def post_generate(base_url, payload, timeout):
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        base_url + "/api/generate", data=body,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    start = time.perf_counter()
    first_text = None
    pieces = []
    final = None
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for line in response:
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("error"):
                raise RuntimeError(event["error"])
            chunk = event.get("response", "")
            if chunk:
                if first_text is None:
                    first_text = time.perf_counter() - start
                pieces.append(chunk)
            if event.get("done"):
                final = event
    elapsed = time.perf_counter() - start
    if final is None:
        raise RuntimeError("Ollama stream ended without a final done event")
    return {
        "response": "".join(pieces),
        "first_text_s": first_text,
        "wall_latency_s": elapsed,
        "ollama_total_s": ns_to_s(final.get("total_duration")),
        "load_s": ns_to_s(final.get("load_duration")),
        "prompt_tokens": final.get("prompt_eval_count"),
        "output_tokens": final.get("eval_count"),
        "generation_s": ns_to_s(final.get("eval_duration")),
        "ollama_final": final,
    }


def ns_to_s(value):
    return value / 1_000_000_000 if isinstance(value, (int, float)) else None


def model_metadata(base_url, model):
    tags = get_json(base_url, "/api/tags")
    for item in tags.get("models", []):
        if model in (item.get("name"), item.get("model")):
            if item.get("remote_host") or item.get("remote_model"):
                raise ValueError(f"Model {model!r} is remote; this harness requires a local model")
            return {"digest": item.get("digest"), "size_bytes": item.get("size"),
                    "details": item.get("details", {})}
    available = ", ".join(x.get("name", "?") for x in tags.get("models", []))
    raise ValueError(f"Model {model!r} is not installed. Available: {available or '(none)'}")


def load_prompts(path, limit):
    prompts = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("Prompt file must contain a nonempty JSON array")
    ids = [p["id"] for p in prompts]
    if len(ids) != len(set(ids)):
        raise ValueError("Prompt IDs must be unique")
    if limit is not None and limit < 1:
        raise ValueError("--limit must be at least 1")
    return prompts[:limit] if limit else prompts


def run(args):
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1")
    base_url = args.base_url.rstrip("/")
    prompts = load_prompts(args.prompts, args.limit)
    model_info = model_metadata(base_url, args.model)
    try:
        version = get_json(base_url, "/api/version").get("version")
    except (OSError, ValueError):
        version = None
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    run_id = str(uuid.uuid4())
    common = {
        "run_id": run_id, "model": args.model, "model_info": model_info,
        "ollama_version": version, "base_url": base_url,
        "hardware": collect(args.device_label), "temperature": args.temperature,
        "num_predict": args.num_predict, "seed": args.seed,
        "prompt_set_sha256": hashlib.sha256(Path(args.prompts).read_bytes()).hexdigest(),
        "prompt_count": len(prompts),
    }
    count = 0
    errors = 0

    def request(prompt):
        return post_generate(base_url, {
            "model": args.model, "prompt": prompt, "stream": True,
            "options": {"temperature": args.temperature,
                        "num_predict": args.num_predict, "seed": args.seed},
        }, args.timeout)

    with out.open("a", encoding="utf-8") as handle:
        def measure(prompt, phase, iteration):
            nonlocal count, errors
            record = dict(common, timestamp_utc=datetime.now(timezone.utc).isoformat(),
                          prompt_id=prompt["id"], category=prompt["category"],
                          phase=phase, iteration=iteration)
            try:
                result = request(prompt["prompt"])
                record.update(result)
                record["tokens_per_s"] = (
                    result["output_tokens"] / result["generation_s"]
                    if result["output_tokens"] is not None and result["generation_s"]
                    else None
                )
                record["quality"] = evaluate(result["response"], prompt.get("score"))
            except (OSError, ValueError, RuntimeError, urllib.error.URLError) as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                errors += 1
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            count += 1
            status = "ERROR" if "error" in record else record["quality"]["status"]
            print(f"{count:>3} {phase:<6} {prompt['id']:<22} {status}", flush=True)

        if args.cold:
            # Ollama's empty prompt with keep_alive=0 unloads the model first.
            post_generate(base_url, {"model": args.model, "prompt": "", "keep_alive": 0,
                                     "stream": True}, args.timeout)
            measure(prompts[0], "cold", 0)
        # A single unscored warm-up precedes every measured prompt.
        request("Reply with the word ready.")
        for prompt in prompts:
            for iteration in range(1, args.repeats + 1):
                measure(prompt, "warm", iteration)
    print(f"Saved {count} records ({errors} errors) to {out}")
    return 1 if errors else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="llama3.2:latest")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--prompts", default=str(Path(__file__).with_name("prompts.json")))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None, help="First N prompts, for smoke tests")
    parser.add_argument("--out", default="results/runs.jsonl")
    parser.add_argument("--cold", action="store_true", help="Unload model and measure one cold request")
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--num-predict", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--device-label", help="Your label, e.g. Mac mini M4; accelerator remains unverified")
    args = parser.parse_args()
    try:
        return run(args)
    except (OSError, ValueError, RuntimeError, urllib.error.URLError) as exc:
        print(f"Benchmark setup failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
