#!/usr/bin/env python
"""
结果分析脚本 - 分析数据清洗结果
"""
import json
import argparse
from collections import Counter
from pathlib import Path


def analyze_results(result_file):
    """分析清洗结果"""
    print("=" * 70)
    print(f"分析结果文件: {result_file}")
    print("=" * 70)
    
    # 加载数据
    with open(result_file, 'r', encoding='utf-8') as f:
        if result_file.endswith('.jsonl'):
            data = [json.loads(line) for line in f]
        else:
            data = json.load(f)
    
    total = len(data)
    print(f"\n总数据量: {total}")
    
    # 统计过滤情况
    filtered = [d for d in data if d.get('filtered', False)]
    kept = [d for d in data if not d.get('filtered', False)]
    
    print(f"保留数据: {len(kept)} ({len(kept)/total*100:.2f}%)")
    print(f"过滤数据: {len(filtered)} ({len(filtered)/total*100:.2f}%)")
    
    # 统计分数分布
    scores = [d.get('score', -1) for d in data if d.get('score', -1) >= 0]
    if scores:
        print(f"\n分数统计:")
        print(f"  平均分: {sum(scores)/len(scores):.2f}")
        print(f"  最高分: {max(scores)}")
        print(f"  最低分: {min(scores)}")
        
        # 分数分布
        score_dist = Counter(scores)
        print(f"\n分数分布:")
        for score in sorted(score_dist.keys()):
            count = score_dist[score]
            print(f"  分数 {score}: {count} ({count/len(scores)*100:.2f}%)")
    
    # 统计过滤原因
    if filtered:
        print(f"\n过滤原因统计:")
        filter_reasons = Counter([d.get('filter_reason', 'Unknown') for d in filtered])
        for reason, count in filter_reasons.most_common():
            print(f"  {reason}: {count} ({count/len(filtered)*100:.2f}%)")
    
    # 统计错误
    errors = [d for d in data if 'error' in d]
    if errors:
        print(f"\n错误数据: {len(errors)}")
        error_types = Counter([d.get('error', 'Unknown')[:50] for d in errors])
        for error, count in error_types.most_common(5):
            print(f"  {error}...: {count}")
    
    # 启发式规则统计
    heuristic_filtered = [d for d in filtered if 'heuristic_flags' in d]
    if heuristic_filtered:
        print(f"\n启发式规则过滤: {len(heuristic_filtered)}")
        all_flags = []
        for d in heuristic_filtered:
            all_flags.extend(d.get('heuristic_flags', []))
        flag_counts = Counter(all_flags)
        for flag, count in flag_counts.most_common():
            print(f"  {flag}: {count}")
    
    print("\n" + "=" * 70)


def compare_results(file1, file2):
    """比较两个结果文件"""
    print("=" * 70)
    print(f"比较结果文件")
    print("=" * 70)
    
    # 加载数据
    with open(file1, 'r', encoding='utf-8') as f:
        if file1.endswith('.jsonl'):
            data1 = [json.loads(line) for line in f]
        else:
            data1 = json.load(f)
    
    with open(file2, 'r', encoding='utf-8') as f:
        if file2.endswith('.jsonl'):
            data2 = [json.loads(line) for line in f]
        else:
            data2 = json.load(f)
    
    print(f"\n文件1: {file1}")
    print(f"  数据量: {len(data1)}")
    print(f"  保留: {len([d for d in data1 if not d.get('filtered', False)])}")
    
    print(f"\n文件2: {file2}")
    print(f"  数据量: {len(data2)}")
    print(f"  保留: {len([d for d in data2 if not d.get('filtered', False)])}")
    
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="分析数据清洗结果")
    parser.add_argument('result_file', help='结果文件路径')
    parser.add_argument('--compare', help='比较的第二个文件（可选）')
    
    args = parser.parse_args()
    
    analyze_results(args.result_file)
    
    if args.compare:
        compare_results(args.result_file, args.compare)


if __name__ == "__main__":
    main()

