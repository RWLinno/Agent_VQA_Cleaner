#!/bin/bash

# AgentCleaner - 3D数据一键启动脚本（支持深度图）

echo "========================================"
echo "AgentCleaner - VQA数据清洗系统 (3D数据)"
echo "========================================"

# 设置GPU（根据需要修改）
export CUDA_VISIBLE_DEVICES=0,1
echo "使用GPU: $CUDA_VISIBLE_DEVICES"

# 设置并发数（根据GPU数量调整）
export NUM_PROCESSORS=8
export BATCH_SIZE=4

# ========================================
# 数据路径配置（RefSpatial 3D数据）
# ========================================
# 重要：ROOT_PATH 应指向 3D/ 目录（不是 3D/image/）
#       系统会自动在 image/ 和 depth/ 子目录中查找文件
BASE_3D="/mnt/data/datasets/transferred_datasets/datasets_train/refspatial/3D"

# 【配置1】标准选择题（默认）
JSON_FILE="$BASE_3D/choice_qa_3D_qwenvl.json"
OUTPUT_FILE="./output/3D/choice_qa_cleaned.jsonl"

# 【配置2】推理模板问题 - 取消下面注释使用
# JSON_FILE="$BASE_3D/reasoning_template_qa_3D_qwenvl.json"
# OUTPUT_FILE="./output/3D/reasoning_template_qa_cleaned.jsonl"

# 【配置3】视觉选择题 - 取消下面注释使用
# JSON_FILE="$BASE_3D/visual_choice_qa_3D_qwenvl.json"
# OUTPUT_FILE="./output/3D/visual_choice_qa_cleaned.jsonl"

# 【配置4】空置检测问题 - 取消下面注释使用
# JSON_FILE="$BASE_3D/vacant_qa_3D_qwenvl.json"
# OUTPUT_FILE="./output/3D/vacant_qa_cleaned.jsonl"

# ROOT_PATH 统一指向 3D 根目录
# 注意：应该指向包含 images/ 和 depth/ 目录的父目录
ROOT_PATH="$BASE_3D/"

# 创建输出目录
mkdir -p ./output/3D
mkdir -p ./logs

# 日志文件
LOG_FILE="./logs/cleaning_3d_$(date +%Y%m%d_%H%M%S).log"

echo "数据源: $JSON_FILE"
echo "根路径: $ROOT_PATH"
echo "输出文件: $OUTPUT_FILE"
echo "日志文件: $LOG_FILE"
echo ""
echo "提示: 系统会自动在以下子目录中查找文件："
echo "  - RGB图像: image/ 或 image_multi_view/ 或 image_visual_choice/"
echo "  - 深度图: depth/ 或 depth_multi_view/"
echo "========================================"

# 检查文件是否存在
if [ ! -f "$JSON_FILE" ]; then
    echo "错误: 输入文件不存在: $JSON_FILE"
    echo "请修改 quick_start_3d.sh 中的 JSON_FILE 路径"
    exit 1
fi

if [ ! -d "$ROOT_PATH" ]; then
    echo "错误: 图像根目录不存在: $ROOT_PATH"
    echo "请修改 quick_start_3d.sh 中的 ROOT_PATH 路径"
    exit 1
fi

# 启动数据清洗（3D数据会自动检测并处理深度图）
echo "开始清洗3D数据（包含RGB和深度图）..."
python main.py \
    --json_file "$JSON_FILE" \
    --root_path "$ROOT_PATH" \
    --output "$OUTPUT_FILE" \
    --num_processors $NUM_PROCESSORS \
    --batch_size $BATCH_SIZE \
    --log_level INFO \
    --log_file "$LOG_FILE" \
    "$@"

# 检查执行结果
if [ $? -eq 0 ]; then
    echo "========================================"
    echo "3D数据清洗完成!"
    echo "结果文件: $OUTPUT_FILE"
    echo "清洗后数据: ${OUTPUT_FILE/.jsonl/_clean.jsonl}"
    echo "完整结果: ${OUTPUT_FILE/.jsonl/_full.jsonl}"
    echo "日志文件: $LOG_FILE"
    echo "========================================"
else
    echo "========================================"
    echo "数据清洗失败，请查看日志: $LOG_FILE"
    echo "========================================"
    exit 1
fi


