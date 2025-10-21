#!/bin/bash

# AgentCleaner 一键启动脚本
# 支持2D和3D数据（自动检测并处理深度图）

echo "========================================"
echo "AgentCleaner - VQA数据清洗系统"
echo "========================================"

# 设置GPU（根据需要修改）
export CUDA_VISIBLE_DEVICES=0,1,2,3
echo "使用GPU: $CUDA_VISIBLE_DEVICES"

# 设置并发数（根据GPU数量调整）
export NUM_PROCESSORS=8
export BATCH_SIZE=4

# ========================================
# 数据路径配置（请根据实际情况修改）
# ========================================

# 【2D数据示例】
JSON_FILE="/path/to/your/dataset/2D/choice_qa.json"
ROOT_PATH="/path/to/your/dataset/2D/image/"
OUTPUT_FILE="./output/2D/choice_qa_cleaned.jsonl"

# 【3D数据示例 - 取消下面的注释来使用】
# 支持的3D数据类型：
#   - choice_qa.json: 标准选择题（单图+单深度图）
#   - multi_view_qa.json: 多视角问题（多图+多深度图）
#   - visual_choice_qa.json: 视觉选择题（带边界框）
#   - reasoning_template_qa.json: 推理模板问题
#   - vacant_qa.json: 空置检测问题
# JSON_FILE="/path/to/your/dataset/3D/choice_qa.json"
# ROOT_PATH="/path/to/your/dataset/3D/"
# OUTPUT_FILE="./output/3D/choice_qa_cleaned.jsonl"

# 创建输出目录（自动根据OUTPUT_FILE创建）
OUTPUT_DIR=$(dirname "$OUTPUT_FILE")
mkdir -p "$OUTPUT_DIR"
mkdir -p ./logs

# 日志文件
LOG_FILE="./logs/cleaning_$(date +%Y%m%d_%H%M%S).log"

echo "数据源: $JSON_FILE"
echo "图像路径: $ROOT_PATH"
echo "输出文件: $OUTPUT_FILE"
echo "日志文件: $LOG_FILE"
echo ""
echo "提示: 系统会自动检测并处理深度图（如果存在）"
echo "========================================"

# 检查文件是否存在
if [ ! -f "$JSON_FILE" ]; then
    echo "错误: 输入文件不存在: $JSON_FILE"
    echo "请修改 quick_start.sh 中的 JSON_FILE 路径"
    exit 1
fi

if [ ! -d "$ROOT_PATH" ]; then
    echo "错误: 图像根目录不存在: $ROOT_PATH"
    echo "请修改 quick_start.sh 中的 ROOT_PATH 路径"
    exit 1
fi

# 启动数据清洗
echo "开始数据清洗..."
echo "注意: 如果数据包含多视角图像或深度图，系统会自动处理"
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
    echo "数据清洗完成!"
    echo "结果文件: $OUTPUT_FILE"
    echo "清洗后数据: ${OUTPUT_FILE/.json/_clean.json}"
    echo "日志文件: $LOG_FILE"
    echo "========================================"
else
    echo "========================================"
    echo "数据清洗失败，请查看日志: $LOG_FILE"
    echo "========================================"
    exit 1
fi

