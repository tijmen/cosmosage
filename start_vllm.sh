python -u -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --model /home/tijmen/cosmosage/models/cosmosage-v3/ \
    --disable-log-requests \
    --dtype bfloat16 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --max-model-len 8192
