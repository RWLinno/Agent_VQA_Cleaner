#!/usr/bin/env python3
"""
处理 ./output/2D/ 目录下所有的.jsonl文件，只保留指定的字段，
并输出到 ./output/2D_ex/ 目录
"""
import json
import os
import glob
from pathlib import Path

def process_jsonl_file(input_file, output_file, fields_to_keep):
    """
    处理单个jsonl文件，只保留指定的字段
    
    Args:
        input_file: 输入文件路径
        output_file: 输出文件路径
        fields_to_keep: 要保留的字段列表
    """
    processed_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line_num, line in enumerate(f_in, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                # 解析JSON
                data = json.loads(line)
                
                # 只保留指定的字段
                filtered_data = {key: data.get(key) for key in fields_to_keep if key in data}
                
                # 写入输出文件
                f_out.write(json.dumps(filtered_data, ensure_ascii=False) + '\n')
                processed_count += 1
                
            except json.JSONDecodeError as e:
                print(f"警告: 文件 {input_file} 第 {line_num} 行解析失败: {e}")
                continue
    
    return processed_count

def main():
    # 要保留的字段
    fields_to_keep = ["question", "answer", "image_path", "filtered", "score"]
    
    # 获取脚本所在目录的父目录（项目根目录）
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    
    # 设置输入和输出目录
    input_dir = project_root / "output" / "2D"
    output_dir = project_root / "output" / "2D_ex"
    
    print(f"正在处理目录: {input_dir}")
    
    # 检查输入目录是否存在
    if not input_dir.exists():
        print(f"错误: 输入目录不存在: {input_dir}")
        return
    
    # 创建输出目录（如果不存在）
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {output_dir}\n")
    
    # 查找所有的.jsonl文件
    jsonl_files = glob.glob(str(input_dir / "*.jsonl"))
    
    if not jsonl_files:
        print(f"未找到任何.jsonl文件在 {input_dir}")
        return
    
    print(f"找到 {len(jsonl_files)} 个.jsonl文件\n")
    
    # 处理每个文件
    for input_file in jsonl_files:
        input_path = Path(input_file)
        # 生成输出文件路径（保持原文件名）
        output_file = output_dir / input_path.name
        
        print(f"处理文件: {input_path.name}")
        
        # 处理文件
        count = process_jsonl_file(input_file, output_file, fields_to_keep)
        
        print(f"  -> 输出文件: {output_file}")
        print(f"  -> 处理了 {count} 条记录\n")
    
    print("所有文件处理完成！")

if __name__ == "__main__":
    main()

