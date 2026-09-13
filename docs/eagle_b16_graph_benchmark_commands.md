# EAGLE B16 Graph

## GSM8K - 启动 vLLM

```bash
export VLLM_PLUGINS=ascend,vspec
export VLLM_USE_AOT_COMPILE=0
export HUST_VSPEC_ENABLED=1
export HUST_VSPEC_METHOD=eagle
export HUST_VSPEC_SHARED_TOKENIZER_PADDING=0
export HUST_VSPEC_USE_MERGED_FULL=0
export HUST_VSPEC_MAX_NUM_SEQS=16
export HUST_VSPEC_EAGLE_TREE_WIDTH=1
export HUST_VSPEC_EAGLE_RELAXED_ACCEPT_TOPK=1
export HUST_VSPEC_CONFIDENCE_ACCEPT_MARGIN=
export HUST_VSPEC_ADAPTIVE_SPECULATION=1
export HUST_VSPEC_ADAPTIVE_POLICY=online
export HUST_VSPEC_ADAPTIVE_MIN_GAMMA=1
export HUST_VSPEC_ADAPTIVE_MAX_GAMMA=4
export HUST_VSPEC_ADAPTIVE_FULL_GRAPH=1
export HUST_VSPEC_ADAPTIVE_ASYNC=1

/usr/local/python3.11.14/bin/vllm serve /model/Qwen2.5-14B-Instruct \
  --served-model-name qwen2.5-14b-eagle-vspec-fp16 \
  --host 127.0.0.1 \
  --port 18180 \
  --dtype float16 \
  --kv-cache-dtype auto \
  --block-size 128 \
  --tensor-parallel-size 1 \
  --pipeline-parallel-size 1 \
  --data-parallel-size 1 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.85 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 8192 \
  --no-enable-prefix-caching \
  --enable-chunked-prefill \
  --no-async-scheduling \
  --no-enforce-eager \
  --seed 0 \
  --scheduling-policy fcfs \
  --distributed-executor-backend mp \
  --disable-custom-all-reduce \
  --no-trust-remote-code \
  --load-format auto \
  --generation-config auto \
  --no-enable-log-requests \
  --uvicorn-log-level info \
  --speculative-config '{"method":"eagle","model":"/model/Eagle-Qwen2.5-14B-Instruct","draft_tensor_parallel_size":1,"num_speculative_tokens":4,"enforce_eager":false}' \
  --compilation-config '{"mode":3,"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --cudagraph-capture-sizes 1 2 3 4 5 6 8 10 12 16 20 24 32 40 48 64 80
```
## GSM8K - 执行压测

```bash
/usr/local/python3.11.14/bin/vllm bench serve \
  --backend openai-chat \
  --base-url http://127.0.0.1:18180 \
  --endpoint /v1/chat/completions \
  --model qwen2.5-14b-eagle-vspec-fp16 \
  --tokenizer /model/Qwen2.5-14B-Instruct \
  --dataset-name custom \
  --dataset-path /run_dir/materialized.jsonl \
  --disable-shuffle \
  --custom-output-len 256 \
  --num-prompts 200 \
  --request-rate inf \
  --temperature 0 \
  --seed 0 \
  --save-result \
  --result-dir /run_dir/benchmark_results/GSM8K-eagle-vspec \
  --result-filename GSM8K.json
```

## ARC-Easy - 启动 vLLM

```bash
export VLLM_PLUGINS=ascend,vspec
export VLLM_USE_AOT_COMPILE=0
export HUST_VSPEC_ENABLED=1
export HUST_VSPEC_METHOD=eagle
export HUST_VSPEC_SHARED_TOKENIZER_PADDING=0
export HUST_VSPEC_USE_MERGED_FULL=0
export HUST_VSPEC_MAX_NUM_SEQS=16
export HUST_VSPEC_EAGLE_TREE_WIDTH=1
export HUST_VSPEC_EAGLE_RELAXED_ACCEPT_TOPK=1
export HUST_VSPEC_CONFIDENCE_ACCEPT_MARGIN=
export HUST_VSPEC_ADAPTIVE_SPECULATION=1
export HUST_VSPEC_ADAPTIVE_POLICY=online
export HUST_VSPEC_ADAPTIVE_MIN_GAMMA=1
export HUST_VSPEC_ADAPTIVE_MAX_GAMMA=4
export HUST_VSPEC_ADAPTIVE_FULL_GRAPH=1
export HUST_VSPEC_ADAPTIVE_ASYNC=1

/usr/local/python3.11.14/bin/vllm serve /model/Qwen2.5-14B-Instruct \
  --served-model-name qwen2.5-14b-eagle-vspec-fp16 \
  --host 127.0.0.1 \
  --port 18180 \
  --dtype float16 \
  --kv-cache-dtype auto \
  --block-size 128 \
  --tensor-parallel-size 1 \
  --pipeline-parallel-size 1 \
  --data-parallel-size 1 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.85 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 8192 \
  --no-enable-prefix-caching \
  --enable-chunked-prefill \
  --no-async-scheduling \
  --no-enforce-eager \
  --seed 0 \
  --scheduling-policy fcfs \
  --distributed-executor-backend mp \
  --disable-custom-all-reduce \
  --no-trust-remote-code \
  --load-format auto \
  --generation-config auto \
  --no-enable-log-requests \
  --uvicorn-log-level info \
  --speculative-config '{"method":"eagle","model":"/model/Eagle-Qwen2.5-14B-Instruct","draft_tensor_parallel_size":1,"num_speculative_tokens":4,"enforce_eager":false}' \
  --compilation-config '{"mode":3,"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --cudagraph-capture-sizes 1 2 3 4 5 6 8 10 12 16 20 24 32 40 48 64 80
```

## ARC-Easy - 执行压测

```bash
/usr/local/python3.11.14/bin/vllm bench serve \
  --backend openai-chat \
  --base-url http://127.0.0.1:18180 \
  --endpoint /v1/chat/completions \
  --model qwen2.5-14b-eagle-vspec-fp16 \
  --tokenizer /model/Qwen2.5-14B-Instruct \
  --dataset-name custom \
  --dataset-path /run_dir/materialized.jsonl \
  --disable-shuffle \
  --custom-output-len 256 \
  --num-prompts 200 \
  --request-rate inf \
  --temperature 0 \
  --seed 0 \
  --save-result \
  --result-dir /run_dir/benchmark_results/ARC-Easy-eagle-vspec \
  --result-filename ARC-Easy.json
```
