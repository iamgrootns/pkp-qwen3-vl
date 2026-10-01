# Qwen3-VL-8B-Instruct — RAG LLM (serverless)

Serverless replacement for the pod command:

```bash
vllm serve Qwen/Qwen3-VL-8B-Instruct --host 0.0.0.0 --port 8001 --dtype auto \
  --enforce-eager --gpu-memory-utilization 0.95 --max-model-len 32768
```

The worker runs that same `vllm serve` inside the container and forwards each
job to it. Requests and responses are normal OpenAI chat-completions JSON.

## Input

```json
{ "input": { "sample": true } }
```
Or any chat-completions body:
```json
{ "input": { "messages": [{"role": "user", "content": "..."}], "max_tokens": 512, "temperature": 0.2 } }
```
Images work the vLLM way (`{"type": "image_url", "image_url": {"url": "https://..." | "data:image/png;base64,..."}}`).

## Output

The vLLM response unchanged: `choices[0].message.content` is the answer.
`{"error": "..."}` on failure.

## Use it from RAG code (OpenAI SDK, drop-in for the pod)

```python
from openai import OpenAI
client = OpenAI(
    base_url="https://api.runpod.ai/v2/<ENDPOINT_ID>/openai/v1",  # was http://<pod>:8001/v1
    api_key="<RUNPOD_API_KEY>",
)
r = client.chat.completions.create(
    model="Qwen/Qwen3-VL-8B-Instruct",
    messages=[{"role": "user", "content": "Context: ...\n\nQuestion: ..."}],
)
print(r.choices[0].message.content)
```
Use `stream=False` (the default); streaming is not wired up.

Plain RunPod call:
```bash
curl -X POST https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync \
  -H "Authorization: Bearer <RUNPOD_API_KEY>" -H "Content-Type: application/json" \
  -d '{"input": {"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 64}}'
```

## Deploy

```bash
docker build -t your-registry/pkp-qwen3-vl:v1 .
docker push your-registry/pkp-qwen3-vl:v1
# RunPod → Serverless → New Endpoint → this image
```

- GPU: 48GB (A6000 / L40S / A40) or 80GB. A 24GB card fits the weights but has
  almost no KV cache left for a 32k context. If you use one, set `MAX_MODEL_LEN=8192`.
- Container disk: ~40GB (image has the ~17GB weights baked in).
- Execution timeout: 600s+. Cold start (load weights + vLLM warmup) takes 1–3 min.
  Warm requests are just inference. Set Active Workers = 1 if you never want a cold start.
- Optional env: `MAX_MODEL_LEN`, `GPU_MEMORY_UTILIZATION`, `SERVED_MODEL_NAME`.

## Local test (needs a GPU)

```bash
python handler.py --test test_input.json
```
