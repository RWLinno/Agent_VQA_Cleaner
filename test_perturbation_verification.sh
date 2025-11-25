#!/bin/bash

# 点位扰动验证测试脚本
# 用于快速测试新的负样本验证功能

echo "=========================================="
echo "点位扰动验证测试"
echo "=========================================="

# 设置EAS配置
export AGENT_TYPE="eas"
export EAS_BASE_URL=${EAS_BASE_URL:-"http://your-eas-url"}
export EAS_TOKEN=${EAS_TOKEN:-"your-token"}
export EAS_MODEL_NAME=${EAS_MODEL_NAME:-"Qwen3-VL-235B-A22B-Instruct-BF16"}

# 数据路径
DATA_DIR="/mnt/data/datasets/knowin_datasets/washing_machine_1029/processed_washing_machine_data_all"
INPUT_FILE="${DATA_DIR}/train_single_label_float.jsonl"
ROOT_PATH="${DATA_DIR}"

# 输出路径
OUTPUT_DIR="/mnt/users/rwl/Agent_VQA_Cleaner/output/test_perturbation"
mkdir -p "${OUTPUT_DIR}"

OUTPUT_FILE="${OUTPUT_DIR}/verification_results.jsonl"
LOG_FILE="${OUTPUT_DIR}/verification.log"
VIS_FILE="${OUTPUT_DIR}/visualization.html"

# 验证参数（可调整）
NUM_WORKERS=2
DISTANCE_THRESHOLD=50
ENABLE_PERTURBATION=true
PERTURBATION_RANGE=100

# 测试数量（只测试前10条）
START_INDEX=0
END_INDEX=10

echo ""
echo "📋 配置信息:"
echo "  输入文件: ${INPUT_FILE}"
echo "  图片根目录: ${ROOT_PATH}"
echo "  输出目录: ${OUTPUT_DIR}"
echo ""
echo "⚙️ 验证参数:"
echo "  Worker数量: ${NUM_WORKERS}"
echo "  距离阈值: ${DISTANCE_THRESHOLD}px"
echo "  启用扰动验证: ${ENABLE_PERTURBATION}"
echo "  扰动范围: ${PERTURBATION_RANGE}px"
echo "  测试范围: ${START_INDEX}-${END_INDEX}"
echo ""

# Step 1: 运行点位验证
echo "=========================================="
echo "Step 1: 运行点位验证"
echo "=========================================="

python main_point_verification.py \
    --input "${INPUT_FILE}" \
    --root_path "${ROOT_PATH}" \
    --output "${OUTPUT_FILE}" \
    --num_workers ${NUM_WORKERS} \
    --distance_threshold ${DISTANCE_THRESHOLD} \
    --enable_perturbation ${ENABLE_PERTURBATION} \
    --perturbation_range ${PERTURBATION_RANGE} \
    --start ${START_INDEX} \
    --end ${END_INDEX} \
    --log_level INFO \
    --log_file "${LOG_FILE}"

if [ $? -eq 0 ]; then
    echo "✓ 点位验证完成"
else
    echo "✗ 点位验证失败"
    exit 1
fi

echo ""

# Step 2: 生成可视化（所有样本）
echo "=========================================="
echo "Step 2: 生成可视化报告"
echo "=========================================="

echo "生成所有样本的可视化..."
python visualize_point_verification.py \
    --input "${OUTPUT_FILE}" \
    --output "${VIS_FILE}" \
    --sample_size 10 \
    --filter all

if [ $? -eq 0 ]; then
    echo "✓ 可视化生成完成: ${VIS_FILE}"
else
    echo "✗ 可视化生成失败"
fi

echo ""

# Step 3: 生成严格筛选失败样本的可视化
FAIL_VIS_FILE="${OUTPUT_DIR}/visualization_strict_fail.html"
echo "生成严格筛选失败样本的可视化..."
python visualize_point_verification.py \
    --input "${OUTPUT_FILE}" \
    --output "${FAIL_VIS_FILE}" \
    --sample_size 10 \
    --filter strict_fail

if [ $? -eq 0 ]; then
    echo "✓ 失败样本可视化生成完成: ${FAIL_VIS_FILE}"
fi

echo ""

# Step 4: 生成负样本检测失败的可视化
NEG_FAIL_VIS_FILE="${OUTPUT_DIR}/visualization_negative_failed.html"
echo "生成负样本检测失败的可视化..."
python visualize_point_verification.py \
    --input "${OUTPUT_FILE}" \
    --output "${NEG_FAIL_VIS_FILE}" \
    --sample_size 10 \
    --filter negative_failed

if [ $? -eq 0 ]; then
    echo "✓ 负样本失败可视化生成完成: ${NEG_FAIL_VIS_FILE}"
fi

echo ""
echo "=========================================="
echo "✓ 测试完成！"
echo "=========================================="
echo ""
echo "📁 输出文件:"
echo "  验证结果: ${OUTPUT_FILE}"
echo "  日志文件: ${LOG_FILE}"
echo "  所有样本可视化: ${VIS_FILE}"
echo "  严格失败样本: ${FAIL_VIS_FILE}"
echo "  负样本失败样本: ${NEG_FAIL_VIS_FILE}"
echo ""
echo "📊 查看结果:"
echo "  在浏览器中打开HTML文件查看可视化报告"
echo ""

