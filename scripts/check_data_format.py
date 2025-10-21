#!/usr/bin/env python
"""
数据格式检查脚本
检查输入 JSON 文件中的数据格式，特别是 image 字段的类型
"""

import json
import sys

def check_data_format(json_file):
    """检查数据格式"""
    print(f"检查文件: {json_file}")
    print("=" * 60)
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"总数据量: {len(data)}")
    print()
    
    issues = []
    
    for i, item in enumerate(data[:10]):  # 检查前10条
        print(f"数据 #{i}:")
        print(f"  keys: {list(item.keys())}")
        
        # 检查 image 字段
        if 'image' in item:
            image_value = item['image']
            image_type = type(image_value).__name__
            print(f"  image 类型: {image_type}")
            print(f"  image 值: {image_value}")
            
            if isinstance(image_value, list):
                print(f"  ⚠️ 警告: image 是列表类型!")
                issues.append(f"第 {i} 条数据的 image 是列表: {image_value}")
            elif not isinstance(image_value, str):
                print(f"  ⚠️ 警告: image 不是字符串!")
                issues.append(f"第 {i} 条数据的 image 类型错误: {image_type}")
        else:
            print(f"  ⚠️ 警告: 缺少 image 字段!")
            issues.append(f"第 {i} 条数据缺少 image 字段")
        
        # 检查 conversations 字段
        if 'conversations' in item:
            convs = item['conversations']
            print(f"  conversations 数量: {len(convs)}")
        else:
            print(f"  ⚠️ 警告: 缺少 conversations 字段!")
            issues.append(f"第 {i} 条数据缺少 conversations 字段")
        
        print()
    
    print("=" * 60)
    if issues:
        print(f"❌ 发现 {len(issues)} 个问题:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✅ 数据格式检查通过!")
    
    return len(issues) == 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python check_data_format.py <input.json>")
        print("示例: python check_data_format.py examples/choice_qa_test.json")
        sys.exit(1)
    
    json_file = sys.argv[1]
    
    try:
        success = check_data_format(json_file)
        sys.exit(0 if success else 1)
    except FileNotFoundError:
        print(f"❌ 文件不存在: {json_file}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

