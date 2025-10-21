#!/usr/bin/env python
"""
路径诊断脚本
帮助诊断为什么所有图片都找不到
"""

import json
import os
import sys
from collections import Counter

def diagnose_paths(json_file, root_path):
    """诊断路径问题"""
    print("=" * 60)
    print("路径诊断工具")
    print("=" * 60)
    print(f"输入文件: {json_file}")
    print(f"根路径: {root_path}")
    print()
    
    # 检查根路径是否存在
    if not os.path.exists(root_path):
        print(f"❌ 根路径不存在: {root_path}")
        return False
    
    print(f"✓ 根路径存在: {root_path}")
    print()
    
    # 加载数据
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"总数据量: {len(data)}")
    print()
    
    # 检查前几条数据
    print("检查前5条数据的路径:")
    print("-" * 60)
    
    found_count = 0
    not_found_count = 0
    
    for i, item in enumerate(data[:5]):
        image_path = item.get('image', '')
        if isinstance(image_path, list):
            image_path = image_path[0] if image_path else ''
        
        full_path = os.path.join(root_path, image_path)
        exists = os.path.exists(full_path)
        
        status = "✓" if exists else "✗"
        print(f"{status} 数据 #{i}:")
        print(f"  image字段: {image_path}")
        print(f"  完整路径: {full_path}")
        print(f"  文件存在: {exists}")
        
        if exists:
            found_count += 1
        else:
            not_found_count += 1
            # 尝试在根路径下搜索这个文件
            filename = os.path.basename(image_path)
            print(f"  正在搜索文件名: {filename}")
            
            # 在根路径下搜索
            for root, dirs, files in os.walk(root_path):
                if filename in files:
                    real_path = os.path.join(root, filename)
                    relative_path = os.path.relpath(real_path, root_path)
                    print(f"  💡 找到文件: {relative_path}")
                    break
            else:
                print(f"  ⚠️ 未找到文件")
        
        print()
    
    print("=" * 60)
    print(f"检查结果: {found_count} 找到, {not_found_count} 未找到")
    print("=" * 60)
    print()
    
    # 分析子目录结构
    print("分析根路径的子目录结构:")
    print("-" * 60)
    subdirs = []
    for item in os.listdir(root_path):
        item_path = os.path.join(root_path, item)
        if os.path.isdir(item_path):
            subdirs.append(item)
    
    if subdirs:
        print(f"发现 {len(subdirs)} 个子目录:")
        for subdir in subdirs[:10]:  # 只显示前10个
            print(f"  - {subdir}/")
    else:
        print("根路径下没有子目录")
    
    print()
    
    # 提供修复建议
    if not_found_count > 0:
        print("=" * 60)
        print("修复建议:")
        print("=" * 60)
        print()
        print("问题: JSON中的image字段可能不包含正确的子目录路径")
        print()
        print("可能的解决方案:")
        print()
        print("1. 确认JSON中的image字段格式")
        print("   应该是: 'image': 'subdir/filename.jpg'")
        print("   而不是: 'image': 'filename.jpg'")
        print()
        print("2. 如果图片在子目录中，需要在image字段中包含子目录")
        print("   例如: 'image': '2D/79588efe106274f4.jpg'")
        print()
        print("3. 或者调整root_path到包含图片的实际目录")
        print(f"   当前: {root_path}")
        print(f"   可能需要: {root_path}/2D")
        print()
    
    return found_count > 0


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python diagnose_paths.py <json_file> <root_path>")
        print("示例: python diagnose_paths.py data.json /path/to/images")
        sys.exit(1)
    
    json_file = sys.argv[1]
    root_path = sys.argv[2]
    
    try:
        success = diagnose_paths(json_file, root_path)
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

