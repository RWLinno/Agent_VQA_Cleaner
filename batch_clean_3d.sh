#!/bin/bash

# AgentCleaner - 3D数据批量清洗脚本
# 自动处理所有5种3D数据类型

echo "========================================"
echo "AgentCleaner - 3D数据批量清洗"
echo "========================================"

# 设置GPU（根据需要修改）
export CUDA_VISIBLE_DEVICES=0,1,2,3
echo "使用GPU: $CUDA_VISIBLE_DEVICES"

# 设置并发数（根据GPU数量调整）
export NUM_PROCESSORS=8
export BATCH_SIZE=4

# 3D数据根目录
BASE_3D="/path/to/your/dataset/3D"
ROOT_PATH="$BASE_3D/"

# 创建输出目录
mkdir -p ./output/3D
mkdir -p ./logs

echo ""
echo "ROOT_PATH: $ROOT_PATH"
echo "输出目录: ./output/3D/"
echo "日志目录: ./logs/"
echo ""

# 定义所有数据类型
declare -a DATA_TYPES=(
    "choice_qa"
    "multi_view_qa"
    "visual_choice_qa"
    "reasoning_template_qa"
    "vacant_qa"
)

# 数据类型的描述
declare -A DESCRIPTIONS=(
    ["choice_qa"]="标准选择题（单图+深度）"
    ["multi_view_qa"]="多视角问题（多图+多深度）"
    ["visual_choice_qa"]="视觉选择题（带边界框）"
    ["reasoning_template_qa"]="推理模板问题"
    ["vacant_qa"]="空置检测问题"
)

# 记录总体开始时间
TOTAL_START_TIME=$(date +%s)
TOTAL_SUCCESS=0
TOTAL_FAILED=0

echo "========================================"
echo "开始批量处理 ${#DATA_TYPES[@]} 种数据类型"
echo "========================================"
echo ""

# 循环处理每种数据类型
for data_type in "${DATA_TYPES[@]}"; do
    echo "========================================"
    echo "[$((TOTAL_SUCCESS + TOTAL_FAILED + 1))/${#DATA_TYPES[@]}] 处理: $data_type"
    echo "描述: ${DESCRIPTIONS[$data_type]}"
    echo "========================================"
    
    # 设置文件路径
    JSON_FILE="$BASE_3D/${data_type}.json"
    OUTPUT_FILE="./output/3D/${data_type}_cleaned.jsonl"
    LOG_FILE="./logs/${data_type}_$(date +%Y%m%d_%H%M%S).log"
    
    # 检查JSON文件是否存在
    if [ ! -f "$JSON_FILE" ]; then
        echo "⚠️  警告: JSON文件不存在，跳过: $JSON_FILE"
        echo ""
        TOTAL_FAILED=$((TOTAL_FAILED + 1))
        continue
    fi
    
    echo "JSON文件: $JSON_FILE"
    echo "输出文件: $OUTPUT_FILE"
    echo "日志文件: $LOG_FILE"
    echo ""
    
    # 记录单个任务开始时间
    TASK_START_TIME=$(date +%s)
    
    # 针对不同数据类型调整参数
    CURRENT_BATCH_SIZE=$BATCH_SIZE
    CURRENT_PROCESSORS=$NUM_PROCESSORS
    
    # 多视角数据需要更小的batch size
    if [ "$data_type" = "multi_view_qa" ]; then
        CURRENT_BATCH_SIZE=$((BATCH_SIZE / 2))
        if [ $CURRENT_BATCH_SIZE -lt 2 ]; then
            CURRENT_BATCH_SIZE=2
        fi
        echo "📊 多视角数据，调整 BATCH_SIZE: $CURRENT_BATCH_SIZE"
    fi
    
    echo "开始清洗..."
    
    # 运行清洗任务
    python main.py \
        --json_file "$JSON_FILE" \
        --root_path "$ROOT_PATH" \
        --output "$OUTPUT_FILE" \
        --num_processors $CURRENT_PROCESSORS \
        --batch_size $CURRENT_BATCH_SIZE \
        --log_level INFO \
        --log_file "$LOG_FILE" 2>&1
    
    # 检查执行结果
    if [ $? -eq 0 ]; then
        TASK_END_TIME=$(date +%s)
        TASK_DURATION=$((TASK_END_TIME - TASK_START_TIME))
        
        echo ""
        echo "✅ 成功完成: $data_type"
        echo "   耗时: ${TASK_DURATION}秒 ($((TASK_DURATION / 60))分钟)"
        echo "   清洗结果: $OUTPUT_FILE"
        echo "   完整数据: ${OUTPUT_FILE/.jsonl/_full.jsonl}"
        
        # 显示统计信息
        if [ -f "$OUTPUT_FILE" ]; then
            CLEANED_COUNT=$(wc -l < "$OUTPUT_FILE")
            echo "   保留数据: $CLEANED_COUNT 条"
        fi
        
        TOTAL_SUCCESS=$((TOTAL_SUCCESS + 1))
    else
        echo ""
        echo "❌ 处理失败: $data_type"
        echo "   请查看日志: $LOG_FILE"
        TOTAL_FAILED=$((TOTAL_FAILED + 1))
    fi
    
    echo ""
    echo "----------------------------------------"
    echo ""
    
    # 短暂延迟，让GPU稍微休息
    sleep 2
done

# 计算总耗时
TOTAL_END_TIME=$(date +%s)
TOTAL_DURATION=$((TOTAL_END_TIME - TOTAL_START_TIME))

# 打印总结
echo "========================================"
echo "批量处理完成！"
echo "========================================"
echo "总耗时: ${TOTAL_DURATION}秒 ($((TOTAL_DURATION / 60))分钟)"
echo "成功: $TOTAL_SUCCESS/${#DATA_TYPES[@]}"
echo "失败: $TOTAL_FAILED/${#DATA_TYPES[@]}"
echo ""
echo "输出目录: ./output/3D/"
echo "日志目录: ./logs/"
echo ""

# 列出所有生成的文件
echo "生成的文件:"
echo "----------------------------------------"
for data_type in "${DATA_TYPES[@]}"; do
    CLEANED_FILE="./output/3D/${data_type}_cleaned.jsonl"
    FULL_FILE="./output/3D/${data_type}_cleaned_full.jsonl"
    
    if [ -f "$CLEANED_FILE" ]; then
        SIZE=$(du -h "$CLEANED_FILE" | cut -f1)
        COUNT=$(wc -l < "$CLEANED_FILE")
        echo "✓ ${data_type}_cleaned.jsonl ($SIZE, $COUNT 条)"
    else
        echo "✗ ${data_type}_cleaned.jsonl (未生成)"
    fi
done
echo "========================================"
echo ""

# 退出状态
if [ $TOTAL_FAILED -eq 0 ]; then
    echo "🎉 所有数据处理成功！"
    exit 0
else
    echo "⚠️  部分数据处理失败，请检查日志"
    exit 1
fi

