"""Qwen3-VL-8B-Instruct for RAG (RunPod Serverless worker).

Serverless version of the pod command:

  vllm serve Qwen/Qwen3-VL-8B-Instruct --host 0.0.0.0 --port 8001 --dtype auto \
    --enforce-eager --gpu-memory-utilization 0.95 --max-model-len 32768

The worker starts that exact server once (inside the container, on localhost)
and forwards every job to it, so requests and responses are plain vLLM /
OpenAI chat-completions JSON — same as talking to the pod.

Input (job["input"]) — one of:
  {"sample": true}
  {"messages": [...], "max_tokens": 512, "temperature": 0.2, ...}   any chat-completions body
  {"openai_route": "/v1/chat/completions", "openai_input": {...}}   sent by RunPod's
      https://api.runpod.ai/v2/<ENDPOINT_ID>/openai/v1 route (OpenAI SDK drop-in)

Output:
  OpenAI chat.completion JSON ({"choices": [{"message": {"content": ...}}], ...})
  {"error": "..."} on failure (never raises).
"""

import json
import os
import subprocess
import sys
import time

import requests
import runpod

MODEL_PATH = os.getenv("MODEL_PATH", "Qwen/Qwen3-VL-8B-Instruct")
SERVED_NAME = os.getenv("SERVED_MODEL_NAME", "Qwen/Qwen3-VL-8B-Instruct")
PORT = os.getenv("VLLM_PORT", "8001")
BASE = f"http://127.0.0.1:{PORT}"

SAMPLE = {
    "messages": [
        {"role": "system", "content": "Answer only from the context."},
        {
            "role": "user",
            "content": "Context: RunPod Serverless scales GPU workers to zero when idle.\n\n"
            "Question: What happens to workers when the endpoint is idle?",
        },
    ],
    "max_tokens": 128,
    "temperature": 0,
}

# Same flags as the pod command. Host is localhost: only this handler talks to it.
_server = subprocess.Popen([
    "vllm", "serve", MODEL_PATH,
    "--served-model-name", SERVED_NAME,
    "--host", "127.0.0.1",
    "--port", PORT,
    "--dtype", "auto",
    "--enforce-eager",
    "--gpu-memory-utilization", os.getenv("GPU_MEMORY_UTILIZATION", "0.95"),
    "--max-model-len", os.getenv("MAX_MODEL_LEN", "32768"),
])


def wait_ready(timeout=1200):
    """Block until vLLM answers /health (cold start loads ~17GB of weights)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _server.poll() is not None:
            raise RuntimeError(f"vllm serve exited with code {_server.returncode}")
        try:
            if requests.get(f"{BASE}/health", timeout=2).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(2)
    raise RuntimeError("vllm serve did not become ready in time")


def handler(job):
    """RunPod entrypoint: job = {"id": ..., "input": {...}}."""
    try:
        job_input = job["input"]

        route = job_input.get("openai_route") or "/v1/chat/completions"
        if job_input.get("openai_input") is not None:
            body = dict(job_input["openai_input"])
        elif job_input.get("sample"):
            body = dict(SAMPLE)
        else:
            body = dict(job_input)

        if route.endswith("/models"):
            return requests.get(f"{BASE}/v1/models", timeout=30).json()

        if not body.get("messages") and not body.get("prompt"):
            return {"error": "Missing sample, messages or openai_input"}

        body["model"] = SERVED_NAME
        body["stream"] = False  # streaming is not wired up; full answer per job

        response = requests.post(f"{BASE}{route}", json=body, timeout=600)
        if response.status_code != 200:
            return {"error": f"vLLM {response.status_code}: {response.text[:500]}"}
        return response.json()
    except Exception as e:
        print(f"Error: {e}", flush=True)
        return {"error": str(e)}


wait_ready()

# Local test mode: python handler.py --test test_input.json  (needs a GPU)
if "--test" in sys.argv:
    _idx = sys.argv.index("--test")
    _path = sys.argv[_idx + 1] if len(sys.argv) > _idx + 1 else "test_input.json"
    with open(_path) as _f:
        _payload = json.load(_f)
    print(json.dumps(handler(_payload), indent=2, ensure_ascii=False))
    _server.terminate()
    raise SystemExit(0)

# Initialize
runpod.serverless.start({"handler": handler})
