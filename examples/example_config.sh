#!/bin/bash

# 示例配置文件 - RefSpatial 2D数据处理

# ========== 2D Choice QA ==========
echo "处理 2D Choice QA 数据..."
python main.py \
    --json_file /hpc2hdd/JH_DATA/share/yingcongchen/PrivateShareGroup/yingcongchen_datasets/knowin_share/datasets/RefSpatial/2D/choice_qa.json \
    --root_path /hpc2hdd/JH_DATA/share/yingcongchen/PrivateShareGroup/yingcongchen_datasets/knowin_share/datasets/RefSpatial/2D \
    --output ./output/2d_choice_qa_cleaned.jsonl \
    --num_processors 8 \
    --batch_size 4 \
    --log_file ./logs/2d_choice_qa.log

# ========== 2D Reasoning Template QA ==========
echo "处理 2D Reasoning Template QA 数据..."
python main.py \
    --json_file /hpc2hdd/JH_DATA/share/yingcongchen/PrivateShareGroup/yingcongchen_datasets/knowin_share/datasets/RefSpatial/2D/reasoning_template_qa.json \
    --root_path /hpc2hdd/JH_DATA/share/yingcongchen/PrivateShareGroup/yingcongchen_datasets/knowin_share/datasets/RefSpatial/2D \
    --output ./output/2d_reasoning_qa_cleaned.jsonl \
    --num_processors 8 \
    --batch_size 4 \
    --log_file ./logs/2d_reasoning_qa.log

# ========== 3D Choice QA ==========
echo "处理 3D Choice QA 数据..."
python main.py \
    --json_file /hpc2hdd/JH_DATA/share/yingcongchen/PrivateShareGroup/yingcongchen_datasets/knowin_share/datasets/RefSpatial/3D/choice_qa.json \
    --root_path /hpc2hdd/JH_DATA/share/yingcongchen/PrivateShareGroup/yingcongchen_datasets/knowin_share/datasets/RefSpatial/3D \
    --output ./output/3d_choice_qa_cleaned.jsonl \
    --num_processors 8 \
    --batch_size 4 \
    --log_file ./logs/3d_choice_qa.log

echo "所有数据处理完成!"

