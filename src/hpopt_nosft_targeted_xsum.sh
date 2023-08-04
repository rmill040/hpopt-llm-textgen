#!/usr/bin/env bash

SCRIPT=hpopt_targeted.py
DATASET=xsum
SAMPLE_SIZE=0.01
METRIC=bertscore_f1
MAX_EVALS=100
RANDOM_STATE=1718

models=(
    "facebook/opt-1.3b:4"
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