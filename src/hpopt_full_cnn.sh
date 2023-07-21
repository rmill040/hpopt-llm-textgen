#!/usr/bin/env bash

SCRIPT=hpopt_full.py
DATASET=cnn_dailymail
SAMPLE_SIZE=0.01
METRIC=bertscore_f1
MAX_EVALS=100
RANDOM_STATE=1718

models=(
    "opt-125m:16"
    "opt-350m:8"
    "opt-1.3b:4"
    "opt-2.7b:4"
    "opt-6.7b:2"
    "opt-13b:1"
    "opt-30b:1"
)

for model in "${models[@]}"; do
    IFS=":" read -r MODEL_NAME BATCH_SIZE <<< "$model"
    python "$SCRIPT" \
        --model-name "$MODEL_NAME" \
        --batch-size "$BATCH_SIZE" \
        --dataset-name "$DATASET" \
        --sample-size "$SAMPLE_SIZE" \
        --metric "$METRIC" \
        --max-evals "$MAX_EVALS" \
        --random-state "$RANDOM_STATE"
done