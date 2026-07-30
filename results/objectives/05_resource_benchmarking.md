# Objective 5: Inference Time & Resource Usage

**Goal:** Measure inference time and resource usage to evaluate real-time practicality for lightweight deployment.

## Benchmark setup

- Hardware: Windows PC, Python 3.12
- Metrics: Per-sample inference time (mean over batch), model file size, memory delta
- Batch sizes: 1, 10, 100

## Inference time

| Model | 1 sample (mean ms) | 10 samples (mean ms) | 100 samples (mean ms) |
|-------|-------------------|---------------------|----------------------|
| Random Forest | **68.3** | 6.7 | 3.4 |
| XGBoost | **1.1** | 0.2 | 0.1 |
| Deep Learning | — | — | — |

## Model size & memory

| Model | File size (MB) | Memory delta (MB) |
|-------|---------------|-------------------|
| Random Forest | 164.4 | ~0 |
| XGBoost | **3.0** | ~0 |
| Deep Learning | — | — |

## Comparison

| Metric | Random Forest | XGBoost | Improvement |
|--------|--------------|---------|-------------|
| Single-sample inference | 68.3 ms | **1.1 ms** | **60× faster** |
| Model size | 164 MB | **3 MB** | **55× smaller** |
| F1-Score | 0.8927 | 0.8928 | Equivalent |

## Real-time practicality

XGBoost at **1.1 ms per sample** can process:
- ~900 samples/second on a single CPU thread
- A full 82K test set in ~90 seconds
- Suitable for real-time edge deployment

## Relevant code
- `src/evaluation/benchmark.py` — `Benchmarker.benchmark_all()`
- `pipeline.py:219-233`
