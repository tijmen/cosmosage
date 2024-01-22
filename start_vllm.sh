python -u -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 \
    --model /home/tijmen/public_models/TheBloke_Nous-Hermes-2-Yi-34B-GPTQ_gptq-4bit-32g-actorder_True \
    --disable-log-requests \
    --dtype float16