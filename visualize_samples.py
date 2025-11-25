"""
可视化工具 - 抽样展示VQA数据清洗结果
支持在HTML中展示图片、问题、答案和点坐标
"""
import os
import json
import argparse
import re
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple
import base64


def extract_coordinate_points(text: str) -> List[Tuple[float, float]]:
    """
    从文本中提取所有点坐标
    
    Args:
        text: 要检测的文本
        
    Returns:
        点坐标列表 [(x1, y1), (x2, y2), ...]
    """
    pattern = r'\[\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\]'
    matches = re.findall(pattern, text)
    points = [(float(x), float(y)) for x, y in matches]
    return points


def image_to_base64(image_path: str) -> str:
    """
    将图片转换为base64编码
    
    Args:
        image_path: 图片路径
        
    Returns:
        base64编码的字符串
    """
    try:
        with open(image_path, 'rb') as f:
            image_data = f.read()
            base64_str = base64.b64encode(image_data).decode('utf-8')
            # 检测图片格式
            ext = os.path.splitext(image_path)[1].lower()
            if ext == '.png':
                mime_type = 'image/png'
            elif ext in ['.jpg', '.jpeg']:
                mime_type = 'image/jpeg'
            elif ext == '.webp':
                mime_type = 'image/webp'
            else:
                mime_type = 'image/png'
            return f"data:{mime_type};base64,{base64_str}"
    except Exception as e:
        print(f"转换图片失败 {image_path}: {e}")
        return ""


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """
    加载JSONL文件
    
    Args:
        file_path: JSONL文件路径
        
    Returns:
        数据列表
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def sample_data(data: List[Dict[str, Any]], 
                sample_size: int = 50,
                filter_mode: str = 'all') -> List[Dict[str, Any]]:
    """
    从数据中抽样
    
    Args:
        data: 数据列表
        sample_size: 抽样数量
        filter_mode: 过滤模式 ('all', 'filtered', 'kept', 'with_points')
        
    Returns:
        抽样后的数据
    """
    if filter_mode == 'filtered':
        data = [d for d in data if d.get('filtered', False)]
    elif filter_mode == 'kept':
        data = [d for d in data if not d.get('filtered', False)]
    elif filter_mode == 'with_points':
        # 只选择答案中包含点坐标的数据
        data = [d for d in data if extract_coordinate_points(d.get('answer', ''))]
    
    if len(data) <= sample_size:
        return data
    
    return random.sample(data, sample_size)


def generate_html_visualization(
    samples: List[Dict[str, Any]],
    output_html: str,
    root_path: str = None,
    title: str = "VQA数据清洗结果可视化"
):
    """
    生成HTML可视化
    
    Args:
        samples: 样本数据列表
        output_html: 输出HTML文件路径
        root_path: 图片根路径（如果样本中只有相对路径）
        title: HTML标题
    """
    # 统计过滤原因
    from collections import Counter
    filter_reasons = []
    for s in samples:
        if s.get('filtered', False) and s.get('filter_reason'):
            filter_reasons.append(s['filter_reason'])
    
    reason_counter = Counter(filter_reasons)
    top_reasons = reason_counter.most_common(10)  # 只显示前10个最常见的原因
    
    # 按类别统计过滤原因
    reason_categories = {
        '点坐标过多': 0,
        'VLM评分过低': 0,
        '启发式规则触发': 0,
        '图像错误': 0,
        '其他原因': 0
    }
    
    for s in samples:
        if s.get('filtered', False):
            reason = s.get('filter_reason', '')
            if 'coordinate points' in reason or '点坐标' in reason:
                reason_categories['点坐标过多'] += 1
            elif 'VLM rated' in reason or 'VLM score' in reason:
                reason_categories['VLM评分过低'] += 1
            elif 'Quality issues' in reason or '质量问题' in reason:
                reason_categories['启发式规则触发'] += 1
            elif 'Image not found' in reason or 'Depth image not found' in reason:
                reason_categories['图像错误'] += 1
            else:
                reason_categories['其他原因'] += 1
    
    # 统计分数分布
    score_dist = {
        '0-3分': sum(1 for s in samples if 0 <= s.get('score', -1) <= 3),
        '4-5分': sum(1 for s in samples if 4 <= s.get('score', -1) <= 5),
        '6-7分': sum(1 for s in samples if 6 <= s.get('score', -1) <= 7),
        '8-10分': sum(1 for s in samples if 8 <= s.get('score', -1) <= 10),
        '错误(-1分)': sum(1 for s in samples if s.get('score', -1) == -1),
    }
    
    # 计算过滤率
    total_count = len(samples)
    filtered_count = sum(1 for s in samples if s.get('filtered', False))
    kept_count = total_count - filtered_count
    filter_rate = (filtered_count / total_count * 100) if total_count > 0 else 0
    
    html_content = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            text-align: center;
            color: #333;
        }}
        .summary {{
            background: white;
            padding: 20px;
            margin: 20px auto;
            max-width: 1200px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .summary h2 {{
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 15px;
        }}
        .summary h3 {{
            color: #34495e;
            margin-top: 20px;
            margin-bottom: 10px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 15px 0;
        }}
        .stat-card {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            border-left: 4px solid #3498db;
        }}
        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
        }}
        .stat-label {{
            font-size: 14px;
            color: #7f8c8d;
            margin-top: 5px;
        }}
        .reason-list {{
            list-style: none;
            padding: 0;
        }}
        .reason-item {{
            background: #fff3e0;
            padding: 10px;
            margin: 8px 0;
            border-radius: 4px;
            border-left: 4px solid #ff9800;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .reason-text {{
            flex: 1;
            margin-right: 15px;
            font-size: 14px;
        }}
        .reason-count {{
            background: #ff9800;
            color: white;
            padding: 4px 12px;
            border-radius: 12px;
            font-weight: bold;
            font-size: 13px;
        }}
        .sample {{
            background: white;
            margin: 20px auto;
            padding: 20px;
            max-width: 1200px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .sample-header {{
            border-bottom: 2px solid #e0e0e0;
            padding-bottom: 10px;
            margin-bottom: 15px;
        }}
        .sample-id {{
            font-size: 18px;
            font-weight: bold;
            color: #2c3e50;
        }}
        .score {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            margin-left: 10px;
        }}
        .score-high {{
            background-color: #d4edda;
            color: #155724;
        }}
        .score-medium {{
            background-color: #fff3cd;
            color: #856404;
        }}
        .score-low {{
            background-color: #f8d7da;
            color: #721c24;
        }}
        .filtered {{
            background-color: #ffebee;
            color: #c62828;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            margin-left: 10px;
        }}
        .kept {{
            background-color: #e8f5e9;
            color: #2e7d32;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            margin-left: 10px;
        }}
        .image-container {{
            position: relative;
            margin: 20px 0;
            display: inline-block;
        }}
        .image-container img {{
            max-width: 800px;
            width: 100%;
            height: auto;
            border: 2px solid #ddd;
            border-radius: 4px;
        }}
        .canvas-overlay {{
            position: absolute;
            top: 0;
            left: 0;
            pointer-events: none;
        }}
        .text-section {{
            margin: 15px 0;
        }}
        .text-label {{
            font-weight: bold;
            color: #555;
            margin-bottom: 5px;
        }}
        .text-content {{
            background: #f8f9fa;
            padding: 12px;
            border-left: 4px solid #007bff;
            border-radius: 4px;
            white-space: pre-wrap;
            word-wrap: break-word;
        }}
        .filter-reason {{
            background: #fff3e0;
            padding: 15px;
            border-left: 4px solid #ff9800;
            border-radius: 4px;
            margin: 10px 0;
        }}
        .filter-reason-title {{
            font-weight: bold;
            color: #e65100;
            margin-bottom: 8px;
            font-size: 15px;
        }}
        .filter-reason-content {{
            color: #5d4037;
            line-height: 1.6;
        }}
        .model-response {{
            background: #e3f2fd;
            padding: 12px;
            border-left: 4px solid #2196f3;
            border-radius: 4px;
            margin: 10px 0;
            font-size: 14px;
            color: #1565c0;
        }}
        .heuristic-flags {{
            background: #fce4ec;
            padding: 10px;
            border-left: 4px solid #e91e63;
            border-radius: 4px;
            margin: 10px 0;
            font-size: 13px;
        }}
        .point-info {{
            background: #e8eaf6;
            padding: 10px;
            border-left: 4px solid #3f51b5;
            border-radius: 4px;
            margin: 10px 0;
            font-size: 13px;
        }}
        .no-filter-reason {{
            background: #e8f5e9;
            padding: 12px;
            border-left: 4px solid #4caf50;
            border-radius: 4px;
            margin: 10px 0;
            color: #2e7d32;
            font-style: italic;
        }}
        .filter-overview {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 12px;
            margin: 20px auto;
            max-width: 1200px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        }}
        .filter-overview h2 {{
            color: white;
            margin-top: 0;
            font-size: 28px;
            margin-bottom: 20px;
        }}
        .filter-rate {{
            font-size: 48px;
            font-weight: bold;
            text-align: center;
            margin: 20px 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        .filter-subtitle {{
            text-align: center;
            font-size: 18px;
            opacity: 0.9;
            margin-bottom: 25px;
        }}
        .category-stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }}
        .category-card {{
            background: rgba(255,255,255,0.15);
            backdrop-filter: blur(10px);
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.2);
        }}
        .category-value {{
            font-size: 32px;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .category-label {{
            font-size: 13px;
            opacity: 0.9;
        }}
        .quick-nav {{
            background: #fff3e0;
            padding: 15px;
            border-radius: 8px;
            margin: 20px auto;
            max-width: 1200px;
            text-align: center;
            border-left: 4px solid #ff9800;
        }}
        .quick-nav a {{
            display: inline-block;
            margin: 5px;
            padding: 8px 16px;
            background: #ff9800;
            color: white;
            text-decoration: none;
            border-radius: 4px;
            font-size: 14px;
            transition: background 0.3s;
        }}
        .quick-nav a:hover {{
            background: #f57c00;
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    
    <div class="filter-overview">
        <h2>🎯 过滤情况总览</h2>
        <div class="filter-rate">{filter_rate:.1f}%</div>
        <div class="filter-subtitle">
            共 {total_count} 个样本，过滤 {filtered_count} 个，保留 {kept_count} 个
        </div>
        
        <h3 style="color: white; margin-top: 25px; margin-bottom: 15px;">📋 过滤原因分类统计</h3>
        <div class="category-stats">
"""
    
    for category, count in reason_categories.items():
        percentage = (count / filtered_count * 100) if filtered_count > 0 else 0
        html_content += f"""
            <div class="category-card">
                <div class="category-value">{count}</div>
                <div class="category-label">{category}<br>({percentage:.1f}%)</div>
            </div>
"""
    
    html_content += """
        </div>
    </div>
"""
    
    # 添加快速导航
    filtered_indices = [i+1 for i, s in enumerate(samples) if s.get('filtered', False)]
    if filtered_indices:
        html_content += f"""
    <div class="quick-nav">
        <strong>🔗 快速跳转到被过滤的样本：</strong><br>
"""
        for idx in filtered_indices[:20]:  # 只显示前20个
            html_content += f"""
        <a href="#sample-{idx}">#{idx}</a>
"""
        if len(filtered_indices) > 20:
            html_content += f"""
        <span style="color: #666; margin-left: 10px;">...以及其他 {len(filtered_indices) - 20} 个</span>
"""
        html_content += """
    </div>
"""
    
    html_content += f"""
    <div class="summary">
        <h2>📊 数据统计概览</h2>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{total_count}</div>
                <div class="stat-label">总样本数</div>
            </div>
            <div class="stat-card" style="border-left-color: #e74c3c;">
                <div class="stat-value" style="color: #e74c3c;">{filtered_count}</div>
                <div class="stat-label">被过滤 ({filter_rate:.1f}%)</div>
            </div>
            <div class="stat-card" style="border-left-color: #27ae60;">
                <div class="stat-value" style="color: #27ae60;">{kept_count}</div>
                <div class="stat-label">保留 ({100-filter_rate:.1f}%)</div>
            </div>
            <div class="stat-card" style="border-left-color: #9b59b6;">
                <div class="stat-value" style="color: #9b59b6;">{sum(1 for s in samples if extract_coordinate_points(s.get('answer', '')))}</div>
                <div class="stat-label">包含点坐标</div>
            </div>
        </div>
        
        <h3>📈 分数分布</h3>
        <div class="stats-grid">
"""
    
    for score_range, count in score_dist.items():
        percentage = (count / len(samples) * 100) if len(samples) > 0 else 0
        html_content += f"""
            <div class="stat-card">
                <div class="stat-value">{count}</div>
                <div class="stat-label">{score_range} ({percentage:.1f}%)</div>
            </div>
"""
    
    html_content += """
        </div>
"""
    
    if top_reasons:
        html_content += """
        <h3>🔍 主要过滤原因 (Top 10)</h3>
        <ul class="reason-list">
"""
        for reason, count in top_reasons:
            # 截断过长的原因文本
            display_reason = reason[:150] + "..." if len(reason) > 150 else reason
            html_content += f"""
            <li class="reason-item">
                <span class="reason-text">{display_reason}</span>
                <span class="reason-count">{count}</span>
            </li>
"""
        html_content += """
        </ul>
"""
    
    html_content += """
    </div>
"""

    for idx, sample in enumerate(samples, 1):
        score = sample.get('score', -1)
        filtered = sample.get('filtered', False)
        question = sample.get('question', '')
        answer = sample.get('answer', '')
        filter_reason = sample.get('filter_reason', '')
        model_response = sample.get('model_response', sample.get('raw_response', ''))
        heuristic_flags = sample.get('heuristic_flags', [])
        original_score = sample.get('original_score', score)
        
        # 提取点坐标
        points = extract_coordinate_points(answer)
        
        # 确定分数样式
        if score >= 7:
            score_class = "score-high"
        elif score >= 5:
            score_class = "score-medium"
        else:
            score_class = "score-low"
        
        # 获取图片路径
        image_path = sample.get('full_image_path', '')
        if not image_path or not os.path.exists(image_path):
            # 尝试使用相对路径 + root_path
            rel_path = sample.get('image_path', '')
            if root_path and rel_path:
                image_path = os.path.join(root_path, rel_path)
        
        # 转换图片为base64
        image_data = ""
        image_width = 800
        image_height = 600
        if image_path and os.path.exists(image_path):
            image_data = image_to_base64(image_path)
            # 尝试获取实际图片尺寸
            try:
                from PIL import Image
                with Image.open(image_path) as img:
                    orig_width, orig_height = img.size
                    # 按比例缩放
                    if orig_width > image_width:
                        image_height = int(orig_height * image_width / orig_width)
                    else:
                        image_width = orig_width
                        image_height = orig_height
            except:
                pass
        
        html_content += f"""
    <div class="sample" id="sample-{idx}">
        <div class="sample-header">
            <span class="sample-id">样本 #{idx}</span>
            <span class="score {score_class}">分数: {score}/10</span>
"""
        
        if original_score != score:
            html_content += f"""
            <span class="score score-medium">原始分数: {original_score}/10</span>
"""
        
        html_content += f"""
            <span class="{'filtered' if filtered else 'kept'}">{'已过滤' if filtered else '已保留'}</span>
        </div>
"""
        
        # 显示图片和点
        if image_data:
            html_content += f"""
        <div class="image-container" id="container-{idx}">
            <img src="{image_data}" id="image-{idx}" alt="Sample Image" 
                 style="max-width: {image_width}px;">
            <canvas class="canvas-overlay" id="canvas-{idx}"></canvas>
        </div>
        <script>
            (function() {{
                var img = document.getElementById('image-{idx}');
                var canvas = document.getElementById('canvas-{idx}');
                var ctx = canvas.getContext('2d');
                
                img.onload = function() {{
                    var displayWidth = img.clientWidth;
                    var displayHeight = img.clientHeight;
                    canvas.width = displayWidth;
                    canvas.height = displayHeight;
                    
                    // 绘制点坐标
                    var points = {json.dumps(points)};
                    points.forEach(function(point, idx) {{
                        var x = point[0];
                        var y = point[1];
                        
                        // 绘制点
                        ctx.beginPath();
                        ctx.arc(x, y, 5, 0, 2 * Math.PI);
                        ctx.fillStyle = 'red';
                        ctx.fill();
                        ctx.strokeStyle = 'white';
                        ctx.lineWidth = 2;
                        ctx.stroke();
                        
                        // 绘制标签
                        ctx.fillStyle = 'white';
                        ctx.strokeStyle = 'black';
                        ctx.lineWidth = 3;
                        ctx.font = 'bold 14px Arial';
                        var label = 'P' + (idx + 1);
                        ctx.strokeText(label, x + 8, y - 8);
                        ctx.fillText(label, x + 8, y - 8);
                    }});
                }};
            }})();
        </script>
"""
        
        # 显示问题
        html_content += f"""
        <div class="text-section">
            <div class="text-label">问题:</div>
            <div class="text-content">{question}</div>
        </div>
"""
        
        # 显示答案
        html_content += f"""
        <div class="text-section">
            <div class="text-label">答案:</div>
            <div class="text-content">{answer}</div>
        </div>
"""
        
        # 显示点坐标信息
        if points:
            points_str = ", ".join([f"P{i+1}:[{x:.2f}, {y:.2f}]" for i, (x, y) in enumerate(points)])
            html_content += f"""
        <div class="point-info">
            <strong>检测到 {len(points)} 个点坐标:</strong> {points_str}
        </div>
"""
        
        # 显示模型响应
        if model_response:
            html_content += f"""
        <div class="model-response">
            <strong>模型评估响应:</strong> {model_response}
        </div>
"""
        
        # 显示启发式标记
        if heuristic_flags:
            flags_str = "; ".join(heuristic_flags)
            html_content += f"""
        <div class="heuristic-flags">
            <strong>质量问题:</strong> {flags_str}
        </div>
"""
        
        # 显示过滤原因（对于所有样本）
        if filtered:
            if filter_reason:
                html_content += f"""
        <div class="filter-reason">
            <div class="filter-reason-title">❌ 过滤原因</div>
            <div class="filter-reason-content">{filter_reason}</div>
        </div>
"""
            else:
                html_content += f"""
        <div class="filter-reason">
            <div class="filter-reason-title">❌ 过滤原因</div>
            <div class="filter-reason-content">分数低于阈值（未提供详细原因）</div>
        </div>
"""
        else:
            # 对于未被过滤的样本，也显示一个提示
            html_content += f"""
        <div class="no-filter-reason">
            ✅ 此样本已通过筛选（未被过滤）
        </div>
"""
        
        html_content += """
    </div>
"""
    
    html_content += """
</body>
</html>
"""
    
    # 保存HTML
    os.makedirs(os.path.dirname(output_html) if os.path.dirname(output_html) else '.', exist_ok=True)
    with open(output_html, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✓ 可视化HTML已生成: {output_html}")


def main():
    parser = argparse.ArgumentParser(
        description="可视化VQA数据清洗结果 - 抽样展示并在HTML中绘制点坐标",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 从完整结果中随机抽样50条
  python visualize_samples.py --input output/choice_qa_full.jsonl --output vis.html
  
  # 只可视化被过滤的数据
  python visualize_samples.py --input output/choice_qa_full.jsonl --output vis_filtered.html --filter filtered
  
  # 只可视化包含点坐标的数据
  python visualize_samples.py --input output/choice_qa_full.jsonl --output vis_points.html --filter with_points
  
  # 指定根路径和抽样数量
  python visualize_samples.py --input output/choice_qa_full.jsonl --output vis.html \\
      --root_path /path/to/images --sample_size 100
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='输入JSONL文件路径（通常是*_full.jsonl）'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='输出HTML文件路径'
    )
    parser.add_argument(
        '--root_path',
        type=str,
        default=None,
        help='图片根目录路径（如果JSONL中只有相对路径）'
    )
    parser.add_argument(
        '--sample_size',
        type=int,
        default=50,
        help='抽样数量 (默认: 50)'
    )
    parser.add_argument(
        '--filter',
        type=str,
        choices=['all', 'filtered', 'kept', 'with_points'],
        default='all',
        help='过滤模式: all(全部), filtered(仅被过滤), kept(仅保留), with_points(仅包含点坐标) (默认: all)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子 (默认: 42)'
    )
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    
    # 检查输入文件
    if not os.path.exists(args.input):
        print(f"错误: 输入文件不存在: {args.input}")
        return
    
    print(f"正在加载数据: {args.input}")
    data = load_jsonl(args.input)
    print(f"✓ 加载了 {len(data)} 条数据")
    
    print(f"正在抽样 (模式: {args.filter}, 数量: {args.sample_size})...")
    samples = sample_data(data, args.sample_size, args.filter)
    print(f"✓ 抽样得到 {len(samples)} 条数据")
    
    print("正在生成HTML可视化...")
    generate_html_visualization(
        samples=samples,
        output_html=args.output,
        root_path=args.root_path,
        title=f"VQA数据清洗结果可视化 ({args.filter} 模式)"
    )
    
    print(f"\n完成！请在浏览器中打开: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()

