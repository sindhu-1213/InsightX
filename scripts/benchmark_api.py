"""Benchmark script to launch uvicorn locally and measure endpoint latency over 20 CPU requests."""

import subprocess
import sys
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent


def main():
    port = 8008
    base_url = f"http://127.0.0.1:{port}"
    print(f"Starting uvicorn server on port {port}...")

    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "api.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        max_wait = 20
        start_wait = time.time()
        server_ready = False
        while time.time() - start_wait < max_wait:
            try:
                r = requests.get(f"{base_url}/health", timeout=1.0)
                if r.status_code == 200:
                    server_ready = True
                    break
            except Exception:
                time.sleep(0.5)

        if not server_ready:
            print("Server failed to start within timeout.")
            sys.exit(1)

        print("Uvicorn server is running.")

        # 1. GET /health
        r_health = requests.get(f"{base_url}/health")
        print(f"1. GET /health: status={r_health.status_code}, data={r_health.json()}")

        # 2. GET /samples
        r_samples = requests.get(f"{base_url}/samples")
        samples = r_samples.json()
        print(f"2. GET /samples: status={r_samples.status_code}, sample_count={len(samples)}")

        # 3. GET /samples/{id}/image
        sample_01 = samples[0]["id"]
        r_img = requests.get(f"{base_url}/samples/{sample_01}/image")
        print(
            f"3. GET /samples/{sample_01}/image: status={r_img.status_code}, "
            f"size={len(r_img.content)} bytes"
        )

        # 4. GET /samples/{id}/predict
        r_pred = requests.get(
            f"{base_url}/samples/{sample_01}/predict",
            headers={"X-Forwarded-For": "10.0.0.1"},
        )
        pred_data = r_pred.json()
        print(
            f"4. GET /samples/{sample_01}/predict: status={r_pred.status_code}, "
            f"label='{pred_data.get('finding_label')}', "
            f"prob={pred_data.get('calibrated_probability')}"
        )

        # 5. Measure latency over 20 CPU prediction requests
        print("\nMeasuring inference latency over 20 CPU prediction requests...")
        # Warmup request
        _ = requests.get(
            f"{base_url}/samples/{sample_01}/predict",
            headers={"X-Forwarded-For": "10.0.0.99"},
        )

        latencies_ms = []
        status_codes = []
        for i in range(20):
            client_ip = f"10.0.1.{i+1}"
            t0 = time.perf_counter()
            resp = requests.get(
                f"{base_url}/samples/{sample_01}/predict",
                headers={"X-Forwarded-For": client_ip},
            )
            t1 = time.perf_counter()
            elapsed_ms = (t1 - t0) * 1000.0
            latencies_ms.append(elapsed_ms)
            status_codes.append(resp.status_code)
            print(f"   Request {i+1:02d}: status={resp.status_code}, latency={elapsed_ms:.2f} ms")

        median_latency = sorted(latencies_ms)[len(latencies_ms) // 2]
        mean_latency = sum(latencies_ms) / len(latencies_ms)
        min_latency = min(latencies_ms)
        max_latency = max(latencies_ms)

        print("\n" + "=" * 50)
        print("LATENCY BENCHMARK RESULTS (20 CPU REQUESTS)")
        print("=" * 50)
        print(f"Status codes: all {set(status_codes)}")
        print(f"Median latency: {median_latency:.2f} ms")
        print(f"Mean latency:   {mean_latency:.2f} ms")
        print(f"Min latency:    {min_latency:.2f} ms")
        print(f"Max latency:    {max_latency:.2f} ms")
        print("=" * 50)

    finally:
        print("Stopping uvicorn server...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        print("Server shutdown complete.")


if __name__ == "__main__":
    main()
