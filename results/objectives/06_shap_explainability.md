# Objective 6: SHAP Explainability

**Goal:** Add explainability (xAI) using SHAP for every flagged alert — a feature missing from other NIDS implementations.

## Global explanation: summary plot

SHAP TreeExplainer generates a summary plot of top-20 features across the test set. The plot shows:
- **Feature importance** (mean |SHAP value|)
- **Direction of impact** — high/low feature values and their push toward attack vs benign

## Per-sample explanation: waterfall plots

For each flagged attack (top-5 anomalous samples), a waterfall plot visualizes how individual features contribute to the prediction:

- Each sample gets a separate PNG saved to `results/figures/shap_waterfall/`
- Red bars = features pushing toward attack prediction
- Blue bars = features pushing toward benign prediction

## Alert explanation API

The `explain_alert()` method returns a structured dictionary per sample:

```python
{
  "prediction": 1,
  "prediction_score": 1.92,
  "top_features": {
    "sttl": 1.43,
    "smean": 0.45,
    "sbytes": 0.40
  },
  "base_value": 0.80,
  "shap_values": [...]  # full 42-dim array
}
```

This enables per-alert explainability in a production setting — for every detection, the top-3 contributing features and their SHAP values are available.

## Top features across all alerts

| Feature | Mean |SHAP| | Description |
|---------|---------------|-------------|
| `sttl` | Highest | Source-to-destination TTL |
| `ct_state_ttl` | 2nd | Connection state + TTL |
| `ct_dst_sport_ltm` | 3rd | Destination + source port count |

Consistent with RF importance ranking from Objective 3.

## Why SHAP matters

Most NIDS works report just accuracy metrics. SHAP provides:
- **Transparency**: why was this specific flow flagged?
- **Debugging**: identify spurious correlations
- **Trust**: security analysts can verify model reasoning

## Relevant code
- `src/explainability/shap_explainer.py` — `SHAPExplainer`
- `pipeline.py:193-215`
