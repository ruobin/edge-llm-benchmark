# Edge LLM benchmark harness

A small, reproducible Ollama benchmark for comparing local models or machines. It records client-observed latency and Ollama's generation telemetry, scores fixed-answer prompts, and builds a self-contained HTML report. It uses Python's standard library only.

## Quick start

Requirements: Python 3.10+; Ollama with a locally installed model. Start Ollama, then verify `ollama list`. The default model is `llama3.2:latest`; override it with `--model` if needed. The harness rejects Ollama tags marked as remote models. Run commands from this directory.

```bash
python3 -m unittest discover -s tests -v
python3 benchmark.py --model llama3.2:latest --limit 2 --repeats 1 --out results/smoke.jsonl
python3 report.py results/smoke.jsonl --out-dir results/smoke-report
```

Open `results/smoke-report/report.html` locally. The smoke run checks connectivity and output format; it is too small for a performance conclusion.

For a fuller run, keep the same prompt file, options, and repeat count for every model or machine:

```bash
python3 benchmark.py --model llama3.2:latest --repeats 5 --cold \
  --device-label "Machine A" --out results/machine-a.jsonl
python3 benchmark.py --model YOUR_SECOND_MODEL --repeats 5 --cold \
  --device-label "Machine B" --out results/machine-b.jsonl
cat results/machine-a.jsonl results/machine-b.jsonl > results/combined.jsonl
python3 report.py results/combined.jsonl --out-dir results/comparison
```

`--cold` unloads the named model with Ollama's `keep_alive=0` request and records one cold request before warming it and running measured repetitions. It changes that model's cache state, so avoid using it during someone else's Ollama workload. A failed request is saved with an error; the command exits nonzero after recording all attempted requests. Each invocation gets a unique `run_id`; the JSONL output is append-only.
The report groups each `run_id` separately and displays the model digest, generation settings, and prompt-set hash where recorded.

## Metrics and interpretation

| Field | Meaning |
|---|---|
| `first_text_s` | Time from client request to the first nonempty streamed text chunk. This is a **TTFT proxy**, not exact server first-token time. |
| `wall_latency_s` | Client wall time to the final streamed event. |
| `ollama_total_s`, `load_s` | Ollama-reported total and model-load durations. |
| `prompt_tokens`, `output_tokens` | Ollama-reported token counts. |
| `generation_s`, `tokens_per_s` | Ollama's eval duration and output-token rate (`output_tokens / generation_s`). |
| `quality` | Strict pass/fail for 18 fixed-answer prompts, unscored for 2 open-ended prompts. |

Durations in Ollama responses are nanoseconds; this harness converts them to seconds. The report uses median and nearest-rank p95. With five repeats per prompt, p95 is meaningful only across the full set and remains sensitive to prompt mix. Compare the same prompt set, sampling count, output limit, temperature, model version, and workload state. Different models tokenize differently, so tokens/sec is useful but not a completely uniform unit. A single-client sequential run does **not** measure concurrent throughput. The quality score is a small smoke rubric, not a validated model benchmark; inspect failures and review unscored outputs manually. Repeated deterministic prompts do not become independent quality questions. The code never executes generated code.

The `hardware` record identifies OS, architecture, processor, memory where available, and a user label. `accelerator_used` remains `unverified`; this harness does not prove GPU/NPU placement or measure power, peak memory, cost, or energy. To make those claims, add platform telemetry and validate it separately. An Ollama model tag may expose its digest, size, and quantization details; those are saved in each record. Keep the digest constant when comparing hardware.

## Dataset and extension

`prompts.json` contains 20 fixed prompts: arithmetic, extraction, classification, JSON, instruction following, summary, and code. Scoring rules are `exact`, `json`, or omitted. `exact` ignores surrounding whitespace, repeated whitespace, and case; JSON compares the parsed object exactly. To add a task, give it a unique `id`, a `category`, a `prompt`, and an optional `score`. For a serious quality study, use a larger externally validated dataset and a clear scoring rubric.

The code currently targets text generation through Ollama's `/api/generate`. A useful next extension is an adapter interface for Ollama, vLLM, and SGLang, followed by controlled concurrency and standardized hardware telemetry. Keep provider-specific tokenization and timing fields explicit; avoid treating them as interchangeable.

## Dell interview walkthrough

1. **Problem:** Edge-model selection needs latency, output rate, quality, and hardware context together. One headline tokens/sec number can hide cold starts or a slow first response.
2. **Design:** A fixed prompt dataset, repeated sequential runs, streamed response parsing, append-only JSONL records, deterministic scoring, and an HTML/CSV report. The project preserves raw responses and model metadata for audit.
3. **Trade-offs:** First visible text is only a TTFT proxy; prompt mix and tokenizers affect comparisons; automated quality checks cover only tasks with clear answers. The harness labels accelerator use as unverified.
4. **Experiment:** Hold prompts, settings, and model digest fixed across devices. Separate cold from warm results. Compare median and p95, pass rate, and errors. Rerun under controlled conditions before making deployment claims.
5. **Next steps:** Add concurrency/load testing, GPU/NPU and power telemetry, broader quality datasets, and backends for vLLM/SGLang. Present a Pareto view of latency, quality, energy, and cost.

Useful official API references: [Ollama generate endpoint](https://docs.ollama.com/api/generate) and [Ollama API documentation](https://github.com/ollama/ollama/blob/main/docs/api.md).
