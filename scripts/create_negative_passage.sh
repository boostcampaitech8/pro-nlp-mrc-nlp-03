#!/bin/bash

# Negative Passage 데이터셋 생성 및 학습 전체 파이프라인

echo "=========================================="
echo "Step 1: Create Negative Passage Dataset"
echo "=========================================="
echo ""

python ../data_preparation/create_negative_passage_dataset.py \
    --train_dataset_path ../../data/train_dataset \
    --passages_path ../../data/wikipedia_passages_256_128.json \
    --output_path ../../data/train_dataset_negative_passage \
    --top_k_retrieval 100 \
    --rerank_top_k 5 \
    --alpha 0.7 \
    --use_rerank

if [ $? -ne 0 ]; then
    echo "Error: Dataset creation failed!"
    exit 1
fi

echo ""
echo "=========================================="
echo "Step 2: Train Reader Model"
echo "=========================================="
echo ""

# arguments.py의 dataset_name을 negative passage dataset으로 변경
# 또는 command line argument로 전달
python -m training.train \
    --model_name_or_path HANTAEK/klue-roberta-large-korquad-v1-qa-finetuned \
    --dataset_name ../../data/train_dataset_negative_passage \
    --output_dir ../../models/reader_negative_passage \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --save_strategy no \
    --save_total_limit 0 \
    --num_train_epochs 3 \
    --learning_rate 3e-5 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --warmup_steps 500 \
    --weight_decay 0.01 \
    --fp16 \
    --seed 2025 \
    --dataloader_num_workers 4

if [ $? -ne 0 ]; then
    echo "Error: Training failed!"
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ All Done!"
echo "=========================================="
echo "Dataset: ../data/train_dataset_negative_passage"
echo "Model: ./models/reader_negative_passage"
echo "=========================================="
