# Objective 4: Zero-Day Attack Simulation

**Goal:** Simulate zero-day detection via leave-one-attack-out testing — hold out one attack category during training, test on held-out samples, measure detection rate.

## Methodology

For each of 8 attack categories in UNSW-NB15:
1. Remove all samples of category X from training data
2. Train XGBoost on remaining data
3. Evaluate on held-out category X samples (treated as "zero-day" attacks)
4. Report F1, recall, precision

## Results

| Held-Out Attack | Test Samples | Precision | Recall | F1-Score |
|-----------------|-------------|-----------|--------|----------|
| Worms | 174 | 1.000 | 1.000 | **1.000** |
| DoS | 16,353 | 1.000 | 0.994 | **0.997** |
| Exploits | 44,525 | 1.000 | 0.974 | **0.987** |
| Reconnaissance | 13,987 | 1.000 | 0.970 | **0.985** |
| Shellcode | 1,511 | 1.000 | 0.943 | **0.971** |
| Generic | 58,871 | 1.000 | 0.931 | **0.964** |
| Analysis | 2,677 | 1.000 | 0.809 | **0.894** |
| Fuzzers | 24,246 | 1.000 | 0.215 | **0.355** |

## Key findings

- **Worms/DoS/Exploits** are well-detected even when held out — their traffic patterns are distinct enough that the model generalizes
- **Fuzzers** are hardest (F1=0.355) — Fuzzer traffic mimics benign patterns, making zero-day detection extremely challenging
- **Generic** has moderate recall (0.931) — broad category, partially overlaps with known patterns
- **Analysis** has low recall (0.809) due to small sample size (2,677 test samples)

This leave-one-out approach validates that XGBoost generalizes beyond seen attack types, with category-specific variability.

## Relevant code
- `src/evaluation/zero_day.py` — `ZeroDaySimulator.simulate()`
- `pipeline.py:172-190`
