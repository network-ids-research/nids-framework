import logging
import time
import numpy as np
import psutil
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)


class Benchmarker:
    def __init__(self, config: dict):
        self.config = config
        self.inference_repeats = config["benchmark"]["inference_repeats"]
        self.batch_sizes = config["benchmark"]["batch_sizes"]
        self.memory_samples = config["benchmark"]["memory_samples"]

    def measure_inference_time(self, model, X_sample: np.ndarray, model_name: str) -> dict:
        results = {"model": model_name}
        for batch_size in self.batch_sizes:
            if batch_size > len(X_sample):
                continue
            X_batch = X_sample[:batch_size]

            times = []
            for _ in range(self.inference_repeats):
                start = time.perf_counter()
                if hasattr(model, "predict_proba"):
                    _ = model.predict_proba(X_batch)
                else:
                    _ = model.predict(X_batch)
                elapsed = time.perf_counter() - start
                times.append(elapsed)

            times = np.array(times)
            per_sample = times / batch_size
            results[f"batch_{batch_size}_mean_ms"] = np.mean(times) * 1000
            results[f"batch_{batch_size}_std_ms"] = np.std(times) * 1000
            results[f"per_sample_{batch_size}_mean_us"] = np.mean(per_sample) * 1_000_000
            results[f"per_sample_{batch_size}_std_us"] = np.std(per_sample) * 1_000_000

        logger.info(f"Inference time for {model_name}:")
        for key, val in results.items():
            if key != "model":
                logger.info(f"  {key}: {val:.2f}")

        return results

    def measure_memory_usage(self, model, model_name: str) -> dict:
        process = psutil.Process()
        mem_before = process.memory_info().rss / (1024 ** 2)

        mem_samples = []
        for _ in range(self.memory_samples):
            mem_samples.append(process.memory_info().rss / (1024 ** 2))

        mem_after = np.mean(mem_samples)
        model_size = 0

        if hasattr(model, "get_booster"):
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
                model.get_booster().save_model(f.name)
                model_size = Path(f.name).stat().st_size / (1024 ** 2)
        elif hasattr(model, "n_estimators"):
            import joblib
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
                joblib.dump(model, f.name)
                model_size = Path(f.name).stat().st_size / (1024 ** 2)

        return {
            "model": model_name,
            "memory_before_mb": round(mem_before, 2),
            "memory_after_mb": round(mem_after, 2),
            "delta_memory_mb": round(mem_after - mem_before, 2),
            "model_size_mb": round(model_size, 2)
        }

    def benchmark_all(self, models: dict, X_sample: np.ndarray) -> tuple:
        inference_results = []
        memory_results = []

        for name, model in models.items():
            logger.info(f"Benchmarking {name}...")
            inf = self.measure_inference_time(model, X_sample, name)
            mem = self.measure_memory_usage(model, name)
            inference_results.append(inf)
            memory_results.append(mem)

        inf_df = pd.DataFrame(inference_results)
        mem_df = pd.DataFrame(memory_results)

        return inf_df, mem_df
