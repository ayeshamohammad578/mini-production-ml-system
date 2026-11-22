"""Empirical latency / throughput profiling and robustness probes.

Run:  python -m serving.benchmark            (starts its own uvicorn server on :8765)
      python -m serving.benchmark --url http://localhost:8000   (profile a running server)

Measures
  1. Single-record /predict latency, 1,000 sequential requests -> p50/p95/p99, req/s
  2. Concurrent load: 8 client threads x 250 requests          -> p50/p95/p99, req/s
  3. /predict/batch with 100 records per call, 100 calls       -> latency, records/s
  4. Model-only latency (in-process, no HTTP) for 1 record and 1,000 records
  5. Batch CLI throughput on the full 22,500-row test file
  6. Robustness probes: malformed / missing / outlier payloads -> expected status codes
Writes reports/benchmark.json and reports/figures/latency_hist.png
"""
import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import numpy as np
import pandas as pd

from src.config import DATA_DIR, REPORT_DIR, FIG_DIR, RAW_FEATURES

PORT = 8765


def pct(a):
    a = np.asarray(a)
    return {"n": int(len(a)), "p50_ms": float(np.percentile(a, 50)), "p95_ms": float(np.percentile(a, 95)),
            "p99_ms": float(np.percentile(a, 99)), "mean_ms": float(a.mean()), "max_ms": float(a.max())}


def payloads(n, seed=0):
    df = pd.read_csv(DATA_DIR / "test.csv").sample(n, random_state=seed, replace=True)
    out = []
    for r in df.itertuples(index=False):
        d = {c: getattr(r, c) for c in RAW_FEATURES}
        for c in ["age", "late_30_59", "late_60_89", "late_90", "open_lines", "real_estate_lines", "dependents"]:
            d[c] = None if pd.isna(d[c]) else int(d[c])
        d["monthly_income"] = None if pd.isna(d["monthly_income"]) else float(d["monthly_income"])
        d["age"] = max(d["age"], 18)
        d["applicant_id"] = str(r.applicant_id)
        out.append(d)
    return out


def timed_post(client, path, body):
    t0 = time.perf_counter(); r = client.post(path, json=body); ms = (time.perf_counter() - t0) * 1000
    return ms, r.status_code


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--url", default=None); a = ap.parse_args()
    proc = None
    url = a.url or f"http://127.0.0.1:{PORT}"
    if a.url is None:
        proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "serving.app:app", "--port", str(PORT), "--log-level", "warning"])
        for _ in range(100):
            try:
                if httpx.get(url + "/health", timeout=1).status_code == 200: break
            except Exception: time.sleep(0.2)
    res = {}
    try:
        P = payloads(3000)
        with httpx.Client(base_url=url, timeout=10) as c:
            for b in P[:50]: c.post("/predict", json=b)                       # warm-up
            # 1. sequential
            t0 = time.perf_counter(); lat = [timed_post(c, "/predict", b)[0] for b in P[:1000]]; wall = time.perf_counter() - t0
            res["single_sequential"] = pct(lat) | {"requests_per_sec": 1000 / wall}
            seq_lat = lat
            # 3. batch endpoint
            batches = [{"records": P[i * 100 % 2900:(i * 100 % 2900) + 100]} for i in range(100)]
            t0 = time.perf_counter(); blat = [timed_post(c, "/predict/batch", b)[0] for b in batches]; wall = time.perf_counter() - t0
            res["batch_endpoint_100_records"] = pct(blat) | {"records_per_sec": 100 * 100 / wall}
        # 2. concurrent
        def worker(k):
            with httpx.Client(base_url=url, timeout=10) as c:
                return [timed_post(c, "/predict", b) for b in P[k * 250:(k + 1) * 250]]
        t0 = time.perf_counter()
        with ThreadPoolExecutor(8) as ex: out = [x for chunk in ex.map(worker, range(8)) for x in chunk]
        wall = time.perf_counter() - t0
        res["concurrent_8_clients"] = pct([m for m, _ in out]) | {"requests_per_sec": len(out) / wall,
                                                                  "error_rate": float(np.mean([s != 200 for _, s in out]))}
        # 6. robustness probes
        base = {k: v for k, v in P[0].items()}
        probes = {
            "valid record": (base, 200),
            "monthly_income missing (null)": (base | {"monthly_income": None, "dependents": None}, 200),
            "extreme outlier revol_util=50,000": (base | {"revol_util": 50000.0}, 200),
            "bureau code late_90=98": (base | {"late_90": 98, "late_60_89": 98, "late_30_59": 98}, 200),
            "required field missing (age)": ({k: v for k, v in base.items() if k != "age"}, 422),
            "negative debt_ratio": (base | {"debt_ratio": -1.0}, 422),
            "impossible age=7": (base | {"age": 7}, 422),
            "string in numeric field": (base | {"open_lines": "five"}, 422),
            "unknown extra field": (base | {"credit_score": 700}, 422),
            "empty batch": ({"records": []}, 422),
        }
        rob = {}
        with httpx.Client(base_url=url, timeout=10) as c:
            for name, (body, expect) in probes.items():
                path = "/predict/batch" if "batch" in name else "/predict"
                r = c.post(path, json=body)
                rob[name] = {"expected": expect, "got": r.status_code, "pass": r.status_code == expect,
                             "response": r.json() if r.status_code == 200 else r.json().get("detail")}
            r = c.post("/predict", content=b"{not json", headers={"Content-Type": "application/json"})
            rob["malformed JSON body"] = {"expected": 422, "got": r.status_code, "pass": r.status_code == 422}
            res["server_metrics_endpoint"] = c.get("/metrics").json()
        res["robustness"] = rob
    finally:
        if proc: proc.terminate(); proc.wait(10)

    # 4. model-only latency (no HTTP)
    from serving.model_service import ModelService
    svc = ModelService().load()
    X = pd.read_csv(DATA_DIR / "test.csv")[RAW_FEATURES]
    one = [X.iloc[[i]] for i in range(300)]
    for x in one[:20]: svc.score_frame(x)
    t = []
    for x in one:
        t0 = time.perf_counter(); svc.score_frame(x); t.append((time.perf_counter() - t0) * 1000)
    res["model_only_single_record"] = pct(t)
    t0 = time.perf_counter(); svc.score_frame(X.iloc[:1000]); res["model_only_1000_records_ms"] = (time.perf_counter() - t0) * 1000
    # 5. batch CLI
    from serving.predict_batch import run
    res["batch_cli_test_file"] = run(DATA_DIR / "test.csv", REPORT_DIR / "scored_test.csv", "applicant_id")

    REPORT_DIR.mkdir(exist_ok=True); (REPORT_DIR / "benchmark.json").write_text(json.dumps(res, indent=2, default=str))
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.hist(np.clip(seq_lat, 0, np.percentile(seq_lat, 99.5)), bins=60, color="#1F4E79")
    for q, col in [(50, "#2E7D32"), (95, "#FF8F00"), (99, "#C62828")]:
        v = np.percentile(seq_lat, q); ax.axvline(v, color=col, ls="--", label=f"p{q} = {v:.1f} ms")
    ax.set(title="/predict latency, 1,000 sequential requests", xlabel="ms", ylabel="requests"); ax.legend()
    plt.tight_layout(); FIG_DIR.mkdir(parents=True, exist_ok=True); plt.savefig(FIG_DIR / "latency_hist.png", dpi=160)
    print(json.dumps({k: v for k, v in res.items() if k not in ("robustness", "server_metrics_endpoint")}, indent=2, default=str))
    print("Robustness:", {k: v["pass"] for k, v in res["robustness"].items()})


if __name__ == "__main__":
    main()
