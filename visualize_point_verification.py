"""
点位验证可视化工具
支持同时显示原始点（P1）和模型推理点（P2）的对比
"""
import os
import json
import argparse
import random
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import base64

logger = logging.getLogger(__name__)


def image_to_base64(image_path: str) -> str:
    """将图片转换为base64编码"""
    try:
        with open(image_path, 'rb') as f:
            image_data = f.read()
            base64_str = base64.b64encode(image_data).decode('utf-8')
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
    """加载JSONL文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def sample_data(
    data: List[Dict[str, Any]],
    sample_size: int = 50,
    filter_mode: str = 'all'
) -> List[Dict[str, Any]]:
    """从数据中抽样"""
    if filter_mode == 'matched':
        # 只选择点位匹配的数据
        data = [d for d in data if d.get('point_comparison', {}).get('match', False)]
    elif filter_mode == 'mismatched':
        # 只选择点位不匹配的数据
        data = [d for d in data if not d.get('point_comparison', {}).get('match', True)]
    elif filter_mode == 'correct':
        # 只选择PointQA验证正确的数据
        data = [d for d in data if d.get('pointqa_verification', {}).get('is_correct', False)]
    elif filter_mode == 'incorrect':
        # 只选择PointQA验证不正确的数据
        data = [d for d in data if not d.get('pointqa_verification', {}).get('is_correct', True)]
    elif filter_mode == 'strict_pass':
        # 只选择通过严格筛选的数据
        data = [d for d in data if d.get('strict_filtering', {}).get('pass', False)]
    elif filter_mode == 'strict_fail':
        # 只选择未通过严格筛选的数据
        data = [d for d in data if not d.get('strict_filtering', {}).get('pass', True)]
    elif filter_mode == 'negative_failed':
        # 只选择未能识别负样本的数据
        data = [d for d in data if not d.get('perturbation_verification', {}).get('negative_detected', True)]
    
    if len(data) <= sample_size:
        return data
    
    return random.sample(data, sample_size)


def generate_html_visualization(
    samples: List[Dict[str, Any]],
    output_html: str,
    root_path: str = None,
    title: str = "点位验证可视化"
):
    """生成HTML可视化"""
    # 统计信息
    total_count = len(samples)
    
    # 点位对比统计
    matched_count = sum(1 for s in samples if s.get('point_comparison', {}).get('match', False))
    mismatched_count = total_count - matched_count
    match_rate = (matched_count / total_count * 100) if total_count > 0 else 0
    
    # PointQA验证统计
    pointqa_correct_count = sum(1 for s in samples if s.get('pointqa_verification', {}).get('is_correct', False))
    pointqa_incorrect_count = total_count - pointqa_correct_count
    pointqa_correct_rate = (pointqa_correct_count / total_count * 100) if total_count > 0 else 0
    
    # 负样本检测统计
    negative_detected_count = sum(1 for s in samples if s.get('perturbation_verification', {}).get('negative_detected', False))
    negative_failed_count = total_count - negative_detected_count
    negative_detect_rate = (negative_detected_count / total_count * 100) if total_count > 0 else 0
    
    # 严格筛选统计
    strict_pass_count = sum(1 for s in samples if s.get('strict_filtering', {}).get('pass', False))
    strict_fail_count = total_count - strict_pass_count
    strict_pass_rate = (strict_pass_count / total_count * 100) if total_count > 0 else 0
    
    # 平均距离
    distances = [s.get('point_comparison', {}).get('distance', 0) 
                 for s in samples if s.get('point_comparison', {}).get('distance') is not None]
    avg_distance = sum(distances) / len(distances) if distances else 0
    
    # 平均置信度
    confidences = [s.get('pointqa_verification', {}).get('confidence', 0) for s in samples]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
    
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
        .overview {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 12px;
            margin: 20px auto;
            max-width: 1200px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        }}
        .overview h2 {{
            color: white;
            margin-top: 0;
            font-size: 28px;
            margin-bottom: 20px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .stat-card {{
            background: rgba(255,255,255,0.15);
            backdrop-filter: blur(10px);
            padding: 15px;
            border-radius: 8px;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.2);
        }}
        .stat-value {{
            font-size: 32px;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .stat-label {{
            font-size: 13px;
            opacity: 0.9;
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
        .badge {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            margin-left: 10px;
            font-size: 13px;
        }}
        .badge-match {{
            background-color: #d4edda;
            color: #155724;
        }}
        .badge-mismatch {{
            background-color: #f8d7da;
            color: #721c24;
        }}
        .badge-correct {{
            background-color: #d1ecf1;
            color: #0c5460;
        }}
        .badge-incorrect {{
            background-color: #fff3cd;
            color: #856404;
        }}
        .image-container {{
            position: relative;
            margin: 20px 0;
            display: inline-block;
        }}
        .image-container img {{
            width: 1000px;
            height: 1000px;
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
        .comparison-info {{
            background: #e3f2fd;
            padding: 15px;
            border-left: 4px solid #2196f3;
            border-radius: 4px;
            margin: 10px 0;
        }}
        .verification-info {{
            background: #f3e5f5;
            padding: 15px;
            border-left: 4px solid #9c27b0;
            border-radius: 4px;
            margin: 10px 0;
        }}
        .legend {{
            background: #fff3e0;
            padding: 15px;
            border-radius: 8px;
            margin: 20px auto;
            max-width: 1200px;
            border-left: 4px solid #ff9800;
        }}
        .legend-item {{
            display: inline-block;
            margin-right: 20px;
            margin-bottom: 10px;
        }}
        .legend-color {{
            display: inline-block;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            margin-right: 5px;
            vertical-align: middle;
            border: 2px solid white;
        }}
        .p1-color {{ background-color: red; }}
        .p2-color {{ background-color: #00ff00; }}
        .p1-prime-color {{ background-color: #ff6b00; }}
        .badge-strict-pass {{
            background-color: #c3e6cb;
            color: #155724;
        }}
        .badge-strict-fail {{
            background-color: #f5c6cb;
            color: #721c24;
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    
    <div class="overview">
        <h2>📊 验证结果总览</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{total_count}</div>
                <div class="stat-label">总样本数</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{matched_count}</div>
                <div class="stat-label">点位匹配 ({match_rate:.1f}%)</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{mismatched_count}</div>
                <div class="stat-label">点位不匹配 ({100-match_rate:.1f}%)</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{avg_distance:.1f}px</div>
                <div class="stat-label">平均距离</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{pointqa_correct_count}</div>
                <div class="stat-label">PointQA正确 ({pointqa_correct_rate:.1f}%)</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{pointqa_incorrect_count}</div>
                <div class="stat-label">PointQA不正确 ({100-pointqa_correct_rate:.1f}%)</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{negative_detected_count}</div>
                <div class="stat-label">负样本检测成功 ({negative_detect_rate:.1f}%)</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{negative_failed_count}</div>
                <div class="stat-label">负样本检测失败 ({100-negative_detect_rate:.1f}%)</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{strict_pass_count}</div>
                <div class="stat-label">严格筛选通过 ({strict_pass_rate:.1f}%)</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{strict_fail_count}</div>
                <div class="stat-label">严格筛选失败 ({100-strict_pass_rate:.1f}%)</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{avg_confidence:.1f}/10</div>
                <div class="stat-label">平均PointQA置信度</div>
            </div>
        </div>
    </div>
    
    <div class="legend">
        <strong>🎨 图例说明：</strong><br>
        <div class="legend-item">
            <span class="legend-color p1-color"></span>
            <strong>P1 (红色)</strong>: 原始标注点（正样本）
        </div>
        <div class="legend-item">
            <span class="legend-color p2-color"></span>
            <strong>P2 (绿色)</strong>: VLM推理点
        </div>
        <div class="legend-item">
            <span class="legend-color p1-prime-color"></span>
            <strong>P1' (橙色)</strong>: 扰动点（负样本）
        </div>
        <div style="margin-top: 10px;">
            <strong>验证逻辑：</strong>
            <ul style="margin: 5px 0; padding-left: 20px;">
                <li>P1和P2距离近 → 点位匹配</li>
                <li>PointQA判定P1正确 → 标注准确</li>
                <li>PointQA判定P1'不正确 → 能识别错误</li>
                <li>三者同时满足 → 通过严格筛选</li>
            </ul>
        </div>
    </div>
"""
    
    for idx, sample in enumerate(samples, 1):
        item_id = sample.get('id', f'sample_{idx}')
        question = sample.get('question', '')
        answer = sample.get('answer', '')
        label = sample.get('label', '')
        
        # 点位信息
        p1_info = sample.get('point_original', {})
        p2_info = sample.get('point_inferred', {})
        comparison = sample.get('point_comparison', {})
        verification = sample.get('pointqa_verification', {})
        perturbation = sample.get('perturbation_verification', {})
        strict = sample.get('strict_filtering', {})
        
        p1 = p1_info.get('coordinates', [0, 0])
        p2 = p2_info.get('coordinates')
        p1_prime = perturbation.get('perturbed_point')
        
        distance = comparison.get('distance')
        match = comparison.get('match', False)
        is_correct = verification.get('is_correct', False)
        confidence = verification.get('confidence', 0)
        
        negative_detected = perturbation.get('negative_detected', False)
        perturbed_distance = perturbation.get('distance_from_original')
        strict_pass = strict.get('pass', False)
        fail_reasons = strict.get('fail_reasons', [])
        
        # 获取图像路径
        image_path = sample.get('full_image_path', '')
        
        # 转换图片为base64
        # 确保显示为1000x1000，因为数据都是transform到这个尺寸的
        image_data = ""
        image_width = 1000
        image_height = 1000
        if image_path and os.path.exists(image_path):
            image_data = image_to_base64(image_path)
            try:
                from PIL import Image
                with Image.open(image_path) as img:
                    orig_width, orig_height = img.size
                    # 数据应该已经是1000x1000，但如果不是则记录警告
                    if orig_width != 1000 or orig_height != 1000:
                        logger.warning(f"图片尺寸不是1000x1000: {image_path} ({orig_width}x{orig_height})")
            except:
                pass
        
        html_content += f"""
    <div class="sample" id="sample-{idx}">
        <div class="sample-header">
            <span class="sample-id">样本 #{idx}: {item_id}</span>
            <span class="badge {'badge-match' if match else 'badge-mismatch'}">
                {'✓ 点位匹配' if match else '✗ 点位不匹配'}
            </span>
            <span class="badge {'badge-correct' if is_correct else 'badge-incorrect'}">
                {'✓ PointQA正确' if is_correct else '✗ PointQA不正确'}
            </span>
            <span class="badge {'badge-correct' if negative_detected else 'badge-incorrect'}">
                {'✓ 负样本检测' if negative_detected else '✗ 负样本失败'}
            </span>
            <span class="badge {'badge-strict-pass' if strict_pass else 'badge-strict-fail'}">
                {'⭐ 严格通过' if strict_pass else '⚠ 严格失败'}
            </span>
        </div>
"""
        
        # 显示图片和点
        if image_data:
            points_data = {
                'p1': p1,
                'p2': p2 if p2 else None,
                'p1_prime': p1_prime if p1_prime else None
            }
            
            html_content += f"""
        <div class="image-container" id="container-{idx}">
            <img src="{image_data}" id="image-{idx}" alt="Sample Image" 
                 style="width: {image_width}px; height: {image_height}px;">
            <canvas class="canvas-overlay" id="canvas-{idx}" width="{image_width}" height="{image_height}"></canvas>
        </div>
        <script>
            (function() {{
                var canvas = document.getElementById('canvas-{idx}');
                var ctx = canvas.getContext('2d');
                
                // 图片和坐标都是1000x1000，直接绘制
                var points = {json.dumps(points_data)};
                
                // 绘制P1（原始点，红色）
                if (points.p1) {{
                    ctx.beginPath();
                    ctx.arc(points.p1[0], points.p1[1], 8, 0, 2 * Math.PI);
                    ctx.fillStyle = 'red';
                    ctx.fill();
                    ctx.strokeStyle = 'white';
                    ctx.lineWidth = 3;
                    ctx.stroke();
                    
                    // 标签
                    ctx.fillStyle = 'white';
                    ctx.strokeStyle = 'red';
                    ctx.lineWidth = 4;
                    ctx.font = 'bold 16px Arial';
                    ctx.strokeText('P1', points.p1[0] + 12, points.p1[1] - 12);
                    ctx.fillStyle = 'red';
                    ctx.fillText('P1', points.p1[0] + 12, points.p1[1] - 12);
                }}
                
                // 绘制P2（推理点，绿色）
                if (points.p2) {{
                    ctx.beginPath();
                    ctx.arc(points.p2[0], points.p2[1], 8, 0, 2 * Math.PI);
                    ctx.fillStyle = '#00ff00';
                    ctx.fill();
                    ctx.strokeStyle = 'white';
                    ctx.lineWidth = 3;
                    ctx.stroke();
                    
                    // 标签
                    ctx.fillStyle = 'white';
                    ctx.strokeStyle = '#00aa00';
                    ctx.lineWidth = 4;
                    ctx.font = 'bold 16px Arial';
                    ctx.strokeText('P2', points.p2[0] + 12, points.p2[1] + 20);
                    ctx.fillStyle = '#00ff00';
                    ctx.fillText('P2', points.p2[0] + 12, points.p2[1] + 20);
                    
                    // 绘制P1-P2连线（黄色虚线）
                    if (points.p1) {{
                        ctx.beginPath();
                        ctx.moveTo(points.p1[0], points.p1[1]);
                        ctx.lineTo(points.p2[0], points.p2[1]);
                        ctx.strokeStyle = 'yellow';
                        ctx.lineWidth = 2;
                        ctx.setLineDash([5, 5]);
                        ctx.stroke();
                        ctx.setLineDash([]);
                    }}
                }}
                
                // 绘制P1'（扰动点/负样本，橙色）
                if (points.p1_prime) {{
                    ctx.beginPath();
                    ctx.arc(points.p1_prime[0], points.p1_prime[1], 8, 0, 2 * Math.PI);
                    ctx.fillStyle = '#ff6b00';
                    ctx.fill();
                    ctx.strokeStyle = 'white';
                    ctx.lineWidth = 3;
                    ctx.stroke();
                    
                    // 标签
                    ctx.fillStyle = 'white';
                    ctx.strokeStyle = '#cc5500';
                    ctx.lineWidth = 4;
                    ctx.font = 'bold 16px Arial';
                    ctx.strokeText("P1'", points.p1_prime[0] + 12, points.p1_prime[1] + 35);
                    ctx.fillStyle = '#ff6b00';
                    ctx.fillText("P1'", points.p1_prime[0] + 12, points.p1_prime[1] + 35);
                    
                    // 绘制P1-P1'连线（橙色虚线）
                    if (points.p1) {{
                        ctx.beginPath();
                        ctx.moveTo(points.p1[0], points.p1[1]);
                        ctx.lineTo(points.p1_prime[0], points.p1_prime[1]);
                        ctx.strokeStyle = '#ff6b00';
                        ctx.lineWidth = 2;
                        ctx.setLineDash([3, 3]);
                        ctx.stroke();
                        ctx.setLineDash([]);
                    }}
                }}
            }})();
        </script>
"""
        
        # 显示问题和label
        html_content += f"""
        <div class="text-section">
            <div class="text-label">Label: <span style="color: #e91e63; font-size: 16px;">"{label}"</span></div>
        </div>
        
        <div class="text-section">
            <div class="text-label">问题:</div>
            <div class="text-content">{question}</div>
        </div>
"""
        
        # 显示点位对比信息
        if p2:
            distance_str = f"{distance:.2f}px" if distance is not None else "N/A"
            match_str = "✅ 匹配" if match else "❌ 不匹配"
            html_content += f"""
        <div class="comparison-info">
            <strong>🎯 点位对比:</strong><br>
            • P1 (原始点): [{p1[0]:.2f}, {p1[1]:.2f}]<br>
            • P2 (推理点): [{p2[0]:.2f}, {p2[1]:.2f}]<br>
            • 距离: {distance_str}<br>
            • 阈值: {comparison.get('threshold', 50)}px<br>
            • 结果: {match_str}
        </div>
"""
        else:
            html_content += f"""
        <div class="comparison-info">
            <strong>🎯 点位对比:</strong><br>
            • P1 (原始点): [{p1[0]:.2f}, {p1[1]:.2f}]<br>
            • P2 (推理点): <span style="color: red;">生成失败</span><br>
            • VLM响应: {p2_info.get('vlm_response', 'N/A')[:100]}...
        </div>
"""
        
        # 显示PointQA验证信息（正样本）
        pointqa_resp = verification.get('response', '')
        html_content += f"""
        <div class="verification-info">
            <strong>🔍 PointQA验证 (正样本P1):</strong><br>
            • 结果: {'✅ 正确' if is_correct else '❌ 不正确'}<br>
            • 置信度: {confidence:.1f}/10<br>
            • 模型响应: {pointqa_resp[:200]}{"..." if len(pointqa_resp) > 200 else ""}
        </div>
"""
        
        # 显示扰动点验证信息（负样本）
        if p1_prime:
            perturbed_resp = perturbation.get('response', '')
            perturbed_conf = perturbation.get('confidence', 0)
            perturbed_correct = perturbation.get('is_correct', False)
            
            html_content += f"""
        <div class="verification-info" style="background: #fff3e0; border-left-color: #ff9800;">
            <strong>⚠️ 负样本验证 (扰动点P1'):</strong><br>
            • P1'坐标: [{p1_prime[0]:.2f}, {p1_prime[1]:.2f}]<br>
            • P1-P1'距离: {perturbed_distance:.2f}px<br>
            • PointQA结果: {'❌ 判定为正确（验证失败）' if perturbed_correct else '✅ 判定为不正确（验证成功）'}<br>
            • 置信度: {perturbed_conf:.1f}/10<br>
            • 模型响应: {perturbed_resp[:150]}{"..." if len(perturbed_resp) > 150 else ""}
        </div>
"""
        
        # 显示严格筛选结果
        html_content += f"""
        <div class="comparison-info" style="background: {'#c3e6cb' if strict_pass else '#f5c6cb'}; border-left-color: {'#28a745' if strict_pass else '#dc3545'};">
            <strong>⭐ 严格筛选结果:</strong><br>
            • 最终判定: {'✅ 通过' if strict_pass else '❌ 失败'}<br>
"""
        
        if not strict_pass and fail_reasons:
            html_content += "            • 失败原因:<br>\n"
            for reason in fail_reasons:
                html_content += f"              &nbsp;&nbsp;- {reason}<br>\n"
        
        html_content += """
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
        description="可视化点位验证结果 - 对比原始点和VLM推理点",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 可视化所有样本
  python visualize_point_verification.py --input output/verification_results.jsonl --output vis.html
  
  # 只可视化点位不匹配的样本
  python visualize_point_verification.py --input output/verification_results.jsonl --output vis_mismatch.html --filter mismatched
  
  # 只可视化PointQA验证不正确的样本
  python visualize_point_verification.py --input output/verification_results.jsonl --output vis_incorrect.html --filter incorrect
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='输入JSONL文件路径'
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
        help='图片根目录路径（如果需要）'
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
        choices=['all', 'matched', 'mismatched', 'correct', 'incorrect', 
                 'strict_pass', 'strict_fail', 'negative_failed'],
        default='all',
        help='过滤模式 (默认: all)'
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
        title=f"点位验证可视化 ({args.filter} 模式)"
    )
    
    print(f"\n完成！请在浏览器中打开: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()

