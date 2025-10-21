#!/bin/bash

# AgentCleaner 测试脚本
# 使用examples目录中的测试数据进行验证

echo "========================================"
echo "AgentCleaner 测试验证"
echo "========================================"

# 设置GPU（根据需要修改）
export CUDA_VISIBLE_DEVICES=0
echo "使用GPU: $CUDA_VISIBLE_DEVICES"

# 设置并发数（测试数据少，使用单进程）
export NUM_PROCESSORS=1
export BATCH_SIZE=1

# 测试数据路径
JSON_FILE="./examples/choice_qa_test.json"
ROOT_PATH="./examples"
OUTPUT_FILE="./examples/output/test_result.jsonl"

# 创建输出目录
mkdir -p ./examples/output
mkdir -p ./logs

# 日志文件
LOG_FILE="./logs/test_$(date +%Y%m%d_%H%M%S).log"

echo "测试数据: $JSON_FILE"
echo "图像路径: $ROOT_PATH"
echo "输出文件: $OUTPUT_FILE"
echo "日志文件: $LOG_FILE"
echo "========================================"

# 检查文件是否存在
if [ ! -f "$JSON_FILE" ]; then
    echo "错误: 测试数据文件不存在: $JSON_FILE"
    exit 1
fi

if [ ! -f "$ROOT_PATH/images/00056c76aa3dd52a.jpg" ]; then
    echo "错误: 测试图像不存在"
    exit 1
fi

# 运行测试
python main.py \
    --json_file "$JSON_FILE" \
    --root_path "$ROOT_PATH" \
    --output "$OUTPUT_FILE" \
    --num_processors 1 \
    --batch_size 1 \
    --log_level INFO \
    --log_file "$LOG_FILE" \
    --no_concurrent

# 检查执行结果
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "测试完成!"
    echo "========================================"
    
    # 显示结果
    echo ""
    echo "生成的文件："
    ls -lh "$OUTPUT_FILE" 2>/dev/null
    ls -lh "${OUTPUT_FILE/.jsonl/_full.jsonl}" 2>/dev/null
    ls -lh "${OUTPUT_FILE/.jsonl/_system_prompt.txt}" 2>/dev/null
    
    echo ""
    echo "清洗后的数据（前3行）："
    head -3 "$OUTPUT_FILE" 2>/dev/null | python -m json.tool 2>/dev/null || head -3 "$OUTPUT_FILE"
    
    echo ""
    echo "完整统计信息："
    if [ -f "$OUTPUT_FILE" ]; then
        echo "清洗后数据行数: $(wc -l < "$OUTPUT_FILE")"
    fi
    
    if [ -f "${OUTPUT_FILE/.jsonl/_full.jsonl}" ]; then
        echo "完整结果行数: $(wc -l < "${OUTPUT_FILE/.jsonl/_full.jsonl}")"
    fi
    
    echo ""
    echo "查看完整结果："
    echo "  cat $OUTPUT_FILE"
    echo "  cat ${OUTPUT_FILE/.jsonl/_full.jsonl}"
    echo "  cat ${OUTPUT_FILE/.jsonl/_system_prompt.txt}"
    echo ""
    echo "分析结果："
    echo "  python scripts/analyze_results.py ${OUTPUT_FILE/.jsonl/_full.jsonl}"
    
else
    echo ""
    echo "========================================"
    echo "测试失败，请查看日志: $LOG_FILE"
    echo "========================================"
    exit 1
fi

