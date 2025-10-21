#!/bin/bash

# 批量处理脚本 - 支持多GPU分布式处理
# 将数据分成多份，在不同GPU上并行处理

# 配置
JSON_FILE=$1
ROOT_PATH=$2
OUTPUT_DIR=${3:-./output}
TOTAL_DATA=${4:-10000}  # 总数据量
NUM_GPUS=${5:-4}        # GPU数量

if [ -z "$JSON_FILE" ] || [ -z "$ROOT_PATH" ]; then
    echo "用法: $0 <json_file> <root_path> [output_dir] [total_data] [num_gpus]"
    echo "示例: $0 data.json /path/to/images ./output 10000 4"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"
mkdir -p ./logs

# 计算每个GPU处理的数据量
CHUNK_SIZE=$((TOTAL_DATA / NUM_GPUS))

echo "========================================"
echo "批量分布式处理"
echo "========================================"
echo "数据文件: $JSON_FILE"
echo "图像路径: $ROOT_PATH"
echo "输出目录: $OUTPUT_DIR"
echo "总数据量: $TOTAL_DATA"
echo "GPU数量: $NUM_GPUS"
echo "每GPU数据量: $CHUNK_SIZE"
echo "========================================"

# 启动多个进程
PIDS=()
for ((i=0; i<NUM_GPUS; i++)); do
    START=$((i * CHUNK_SIZE))
    END=$(((i + 1) * CHUNK_SIZE))
    
    # 最后一个进程处理到末尾
    if [ $i -eq $((NUM_GPUS - 1)) ]; then
        END=$TOTAL_DATA
    fi
    
    OUTPUT_FILE="$OUTPUT_DIR/output_gpu${i}_${START}_${END}.jsonl"
    LOG_FILE="./logs/gpu${i}_${START}_${END}.log"
    
    echo "启动GPU $i: 处理数据 [$START, $END)"
    
    CUDA_VISIBLE_DEVICES=$i python main.py \
        --json_file "$JSON_FILE" \
        --root_path "$ROOT_PATH" \
        --output "$OUTPUT_FILE" \
        --start $START \
        --end $END \
        --num_processors 1 \
        --batch_size 4 \
        --log_file "$LOG_FILE" \
        --no_concurrent &
    
    PIDS+=($!)
done

# 等待所有进程完成
echo "等待所有进程完成..."
for PID in "${PIDS[@]}"; do
    wait $PID
done

echo "========================================"
echo "所有处理完成!"
echo "结果文件在: $OUTPUT_DIR"
echo "========================================"

# 合并结果（可选）
echo "合并结果文件..."
python -c "
import json
import glob
import os

output_dir = '$OUTPUT_DIR'
files = sorted(glob.glob(os.path.join(output_dir, 'output_gpu*.jsonl')))

all_data = []
for f in files:
    with open(f, 'r', encoding='utf-8') as file:
        for line in file:
            all_data.append(json.loads(line))
    print(f'加载 {f}: {len([line for line in open(f)])} 条数据')

output_file = os.path.join(output_dir, 'merged_clean.jsonl')
with open(output_file, 'w', encoding='utf-8') as f:
    for item in all_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\\n')

print(f'合并完成: {output_file}')
print(f'总共: {len(all_data)} 条清洗后的数据')
"

echo "合并完成!"

