#!/bin/bash
# 实时监控 AgentCleaner 处理进度

LOG_DIR="logs"
OUTPUT_DIR="output"

echo "=========================================="
echo "AgentCleaner 实时监控"
echo "=========================================="
echo ""

# 查找最新的日志文件
LATEST_LOG=$(ls -t $LOG_DIR/cleaning_*.log 2>/dev/null | head -1)

if [ -z "$LATEST_LOG" ]; then
    echo "❌ 未找到运行日志"
    echo "请先运行: python main.py ..."
    exit 1
fi

echo "监控日志: $LATEST_LOG"
echo ""

# 显示处理统计
echo "【处理统计】"
echo "----------------------------------------"

KEPT_COUNT=$(grep -c "\[KEPT\]" "$LATEST_LOG" 2>/dev/null || echo 0)
FILTERED_COUNT=$(grep -c "\[FILTERED\]" "$LATEST_LOG" 2>/dev/null || echo 0)
ERROR_COUNT=$(grep -c "\[ERROR\]" "$LATEST_LOG" 2>/dev/null || echo 0)
TOTAL=$((KEPT_COUNT + FILTERED_COUNT + ERROR_COUNT))

echo "✓ 保留: $KEPT_COUNT"
echo "⚠ 过滤: $FILTERED_COUNT"
echo "✗ 错误: $ERROR_COUNT"
echo "总计: $TOTAL"

if [ $TOTAL -gt 0 ]; then
    KEPT_PERCENT=$(echo "scale=2; $KEPT_COUNT * 100 / $TOTAL" | bc)
    FILTERED_PERCENT=$(echo "scale=2; $FILTERED_COUNT * 100 / $TOTAL" | bc)
    ERROR_PERCENT=$(echo "scale=2; $ERROR_COUNT * 100 / $TOTAL" | bc)
    
    echo ""
    echo "保留率: ${KEPT_PERCENT}%"
    echo "过滤率: ${FILTERED_PERCENT}%"
    echo "错误率: ${ERROR_PERCENT}%"
fi

echo ""

# 显示临时文件
echo "【中间结果文件】"
echo "----------------------------------------"

TEMP_FILES=$(ls -lh $OUTPUT_DIR/*_temp_*.jsonl 2>/dev/null)
if [ -n "$TEMP_FILES" ]; then
    echo "$TEMP_FILES" | awk '{print $9, "(" $5 ")"}'
    
    TEMP_COUNT=$(cat $OUTPUT_DIR/*_temp_*.jsonl 2>/dev/null | wc -l)
    echo ""
    echo "中间结果总行数: $TEMP_COUNT"
else
    echo "暂无中间结果文件（每100条保存一次）"
fi

echo ""

# 显示最近的处理日志
echo "【最近10条处理日志】"
echo "----------------------------------------"
grep "\[KEPT\]\|\[FILTERED\]\|\[ERROR\]" "$LATEST_LOG" 2>/dev/null | tail -10

echo ""

# 显示处理进度
echo "【处理进度】"
echo "----------------------------------------"
tail -20 "$LATEST_LOG" | grep "已处理" | tail -5

echo ""

# 分析错误原因
if [ $ERROR_COUNT -gt 0 ]; then
    echo "【错误原因分析】"
    echo "----------------------------------------"
    
    # 统计错误类型
    echo "错误类型分布:"
    grep "\[ERROR\]" "$LATEST_LOG" | grep -oP "Reason: [^\"]*" | sort | uniq -c | sort -rn | head -5
    
    echo ""
fi

# 分析分数分布
if [ $TOTAL -gt 10 ]; then
    echo "【分数分布分析】"
    echo "----------------------------------------"
    
    echo "分数统计:"
    grep "Score:" "$LATEST_LOG" | grep -oP "Score: \d+" | awk '{print $2}' | sort -n | uniq -c | sort -k2n
    
    echo ""
fi

echo "=========================================="
echo "实时监控完成"
echo "=========================================="
echo ""
echo "提示:"
echo "  - 使用 tail -f 查看实时日志:"
echo "    tail -f $LATEST_LOG | grep '\[KEPT\]\|\[FILTERED\]\|\[ERROR\]'"
echo ""
echo "  - 查看中间结果:"
echo "    ls -lh $OUTPUT_DIR/*_temp*.jsonl"
echo ""

