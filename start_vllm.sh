python -u -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --disable-log-requests
