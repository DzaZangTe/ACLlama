#!/bin/bash

# 设置参数
JSON_FILE="/linzhihang/zhangyuhao/zhanchen/palalingual-acllama-tester/saved/AudioLLMs_voxceleb_accent_test-1.json"  # 替换为您的实际文件路径
EVAL_TYPE="5-point"  # 或者 "binary"
MAX_WORKERS=8  # 并发 API 请求数
OUTPUT_DIR="./eval_results"

# 运行评估
python testers/eval.py \
    --json_file "${JSON_FILE}" \
    --eval_type "${EVAL_TYPE}" \
    --max_workers ${MAX_WORKERS} \
    --output_dir "${OUTPUT_DIR}"