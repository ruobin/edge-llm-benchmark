# Measured local smoke study — 29 September 2026

This is a single-model demonstration of the harness, not a hardware or model ranking. The local Ollama server reported Metal on an Apple M4. The benchmark record itself leaves `accelerator_used` as `unverified` because the harness does not independently inspect device placement.

## Setup

- Model: `llama3.2:latest`, 3.2B, Q4_K_M; digest prefix `a80c4f17acd5`.
- Prompt set: 20 fixed prompts, five warm repetitions each; 18 prompts have strict automatic checks.
- Generation options: temperature 0, seed 42, maximum 128 output tokens.
- One model unload followed by one measured cold prompt, one warm-up request, then 100 measured warm requests.
- Output: [raw JSONL](results/llama3.2-m4-final.jsonl), [CSV summary](results/llama3.2-m4-final-report/summary.csv), [HTML report](results/llama3.2-m4-final-report/report.html). The HTML report is self-contained, with static SVG charts.

## Results

| Metric | Warm run | Cold request |
|---|---:|---:|
| Requests / transport errors | 100 / 0 | 1 / 0 |
| First visible text, median | 156.3 ms | 1,101.8 ms |
| First visible text, p95 | 228.0 ms | Not meaningful (n=1) |
| Client wall latency, median | 219.5 ms | 1,128.9 ms |
| Client wall latency, p95 | 2,812.6 ms | Not meaningful (n=1) |
| Generation rate, median | 63.7 output tokens/s | 72.9 output tokens/s |
| Strict pass rate | 70/90 scored repetitions (77.8%) | 1/1 |

The two open-ended prompts were unscored. Across distinct fixed-answer prompts, 14 of 18 passed consistently; five repetitions of the same prompt are not five independent quality questions. Failures clustered in one arithmetic item, one extraction item, and two JSON-format items. For example, the model answered the median of 2, 3, and 10 as `7` rather than `3`. It also wrapped one requested JSON object in explanatory prose. This separates factual accuracy from output-format compliance. A punctuation-only response also failed an intentionally strict exact-match rule; inspect raw text before treating every failure as a semantic error.

The cold row has one observation, so its p95 value in the CSV is mathematically the same observation and should not be interpreted as a tail estimate. The warm p95 aggregates prompts with very different output lengths, including code generation, so it describes this particular prompt mix. Repeating a controlled run and measuring memory, power, concurrent requests, and device placement would be necessary before making an edge-deployment recommendation.

## Interview summary

“I built a local Ollama benchmark that captures the first visible streamed text, total client latency, server-reported generation rate, model metadata, and strict task checks. I separated cold and warm runs and saved every response for review. On an Apple M4 with Llama 3.2 3B Q4, the warm median first-text latency was 156 ms and the strict pass rate was 14 of 18 distinct tasks. The failures were more informative than a single speed number: they exposed an arithmetic mistake and JSON-format drift. I would validate GPU usage, add concurrency and power telemetry, and compare the same model digest across target devices before recommending a deployment.”
