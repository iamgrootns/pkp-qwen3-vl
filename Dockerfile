# Official vLLM OpenAI image — same server the pod runs. Qwen3-VL needs vLLM >= 0.11.
FROM vllm/vllm-openai:v0.11.0

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Bake the weights (~17GB) into the image so workers never download at runtime
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-VL-8B-Instruct', local_dir='/models/Qwen3-VL-8B-Instruct')"
ENV MODEL_PATH=/models/Qwen3-VL-8B-Instruct
ENV HF_HUB_OFFLINE=1

COPY . /app

# The base image's ENTRYPOINT is the vLLM server; the handler starts it itself.
ENTRYPOINT []
CMD ["python3", "-u", "handler.py"]
