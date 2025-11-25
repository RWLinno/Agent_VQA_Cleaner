"""
验证raw_data处理的坐标是否正确
可视化10个样本，检查点坐标和图片是否都正确transform到1000*1000像素
"""
import os
import json
import random
import base64
from pathlib import Path
from typing import List, Dict, Any


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """加载JSONL文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


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


def extract_points_from_answer(answer: str) -> List[tuple]:
    """从答案中提取点坐标"""
    import re
    points = []
    try:
        pattern = r'\[\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\]'
        matches = re.findall(pattern, answer)
        for x, y in matches:
            points.append((float(x), float(y)))
    except Exception as e:
        print(f"提取点坐标失败: {e}")
    return points


def verify_image_size(image_path: str) -> tuple:
    """验证图片尺寸"""
    try:
        from PIL import Image
        with Image.open(image_path) as img:
            return img.size
    except Exception as e:
        print(f"读取图片尺寸失败 {image_path}: {e}")
        return (0, 0)


def generate_verification_html(
    samples: List[Dict[str, Any]],
    image_root: str,
    output_html: str
):
    """生成验证HTML"""
    
    html_content = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Raw Data 坐标验证</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            text-align: center;
            color: #333;
        }
        .info-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 12px;
            margin: 20px auto;
            max-width: 1200px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        }
        .info-box h2 {
            margin-top: 0;
            color: white;
        }
        .sample {
            background: white;
            margin: 20px auto;
            padding: 20px;
            max-width: 1200px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .sample-header {
            border-bottom: 2px solid #e0e0e0;
            padding-bottom: 10px;
            margin-bottom: 15px;
        }
        .sample-id {
            font-size: 18px;
            font-weight: bold;
            color: #2c3e50;
        }
        .badge {
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-weight: bold;
            margin-left: 10px;
            font-size: 13px;
        }
        .badge-ok {
            background-color: #d4edda;
            color: #155724;
        }
        .badge-warn {
            background-color: #fff3cd;
            color: #856404;
        }
        .badge-error {
            background-color: #f8d7da;
            color: #721c24;
        }
        .image-container {
            position: relative;
            margin: 20px 0;
            display: inline-block;
            border: 2px solid #ddd;
        }
        .image-container img {
            display: block;
            width: 1000px;
            height: 1000px;
        }
        .canvas-overlay {
            position: absolute;
            top: 0;
            left: 0;
            pointer-events: none;
        }
        .info-section {
            margin: 15px 0;
            padding: 12px;
            background: #f8f9fa;
            border-left: 4px solid #007bff;
            border-radius: 4px;
        }
        .info-label {
            font-weight: bold;
            color: #555;
        }
        .legend {
            background: #fff3e0;
            padding: 15px;
            border-radius: 8px;
            margin: 20px auto;
            max-width: 1200px;
            border-left: 4px solid #ff9800;
        }
        .legend-item {
            display: inline-block;
            margin-right: 20px;
        }
        .legend-color {
            display: inline-block;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            margin-right: 5px;
            vertical-align: middle;
            border: 2px solid white;
            background-color: red;
        }
    </style>
</head>
<body>
    <h1>🔍 Raw Data 坐标验证</h1>
    
    <div class="info-box">
        <h2>📋 验证说明</h2>
        <p>本页面用于验证 train_single_label_float.jsonl 中的点坐标是否正确。</p>
    </div>
    
    <div class="legend">
        <strong>🎨 图例说明：</strong>
        <div class="legend-item">
            <span class="legend-color"></span>
            <strong>红色圆点</strong>: 标注点坐标
        </div>
    </div>
"""
    
    for idx, sample in enumerate(samples, 1):
        # 提取信息
        item_id = sample.get('id', f'sample_{idx}')
        messages = sample.get('messages', [])
        images = sample.get('images', [])
        
        question = ''
        answer = ''
        for msg in messages:
            if msg.get('role') == 'user':
                question = msg.get('content', '').replace('<image>', '').strip()
            elif msg.get('role') == 'assistant':
                answer = msg.get('content', '').strip()
        
        # 获取图片路径
        image_name = images[0] if isinstance(images, list) else images
        image_path = os.path.join(image_root, os.path.basename(image_name))
        
        # 验证图片尺寸
        img_width, img_height = verify_image_size(image_path)
        size_ok = (img_width == 1000 and img_height == 1000)
        
        # 提取点坐标
        points = extract_points_from_answer(answer)
        points_ok = all(0 <= x <= 1000 and 0 <= y <= 1000 for x, y in points)
        
        # 转换图片为base64
        image_data = image_to_base64(image_path) if os.path.exists(image_path) else ""
        
        # 状态标签
        if size_ok and points_ok:
            status_badge = '<span class="badge badge-ok">✓ 验证通过</span>'
        elif not size_ok:
            status_badge = '<span class="badge badge-error"> </span>'
        else:
            status_badge = '<span class="badge badge-warn">⚠ 坐标超出范围</span>'
        
        html_content += f"""
    <div class="sample">
        <div class="sample-header">
            <span class="sample-id">样本 #{idx}: {item_id}</span>
            {status_badge}
        </div>
        
        <div class="info-section">
            <span class="info-label">图片信息:</span> {os.path.basename(image_name)}<br>
            <span class="info-label">图片尺寸:</span> {img_width} × {img_height} 像素
        </div>
        
        <div class="info-section">
            <span class="info-label">问题:</span> {question}
        </div>
        
        <div class="info-section">
            <span class="info-label">标注点坐标:</span> {points}<br>
            <span class="info-label">坐标范围:</span>
            {'<span style="color: green;"> ✓ 在 [0, 1000] 范围内</span>' if points_ok else '<span style="color: red;"> ✗ 超出范围</span>'}
        </div>
"""
        
        if image_data:
            points_json = json.dumps(points)
            html_content += f"""
        <div class="image-container" id="container-{idx}">
            <img src="{image_data}" id="image-{idx}" alt="Sample Image">
            <canvas class="canvas-overlay" id="canvas-{idx}" width="1000" height="1000"></canvas>
        </div>
        <script>
            (function() {{
                var canvas = document.getElementById('canvas-{idx}');
                var ctx = canvas.getContext('2d');
                var points = {points_json};
                
                // 绘制所有点
                points.forEach(function(point, i) {{
                    // 绘制红色圆点
                    ctx.beginPath();
                    ctx.arc(point[0], point[1], 8, 0, 2 * Math.PI);
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
                    var label = 'P' + (i + 1);
                    ctx.strokeText(label, point[0] + 12, point[1] - 12);
                    ctx.fillStyle = 'red';
                    ctx.fillText(label, point[0] + 12, point[1] - 12);
                }});
            }})();
        </script>
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
    
    print(f"✓ 验证HTML已生成: {output_html}")


def main():
    """主函数"""
    # 配置路径
    data_file = "/mnt/data/datasets/knowin_datasets/washing_machine_1029/processed_washing_machine_data_all/train_single_label_float.jsonl"
    image_root = "/mnt/data/datasets/knowin_datasets/washing_machine_1029/processed_washing_machine_data_all/visualization/train"
    output_html = "/mnt/users/rwl/Agent_VQA_Cleaner/output/verify_raw_data.html"
    
    print("=" * 60)
    print("Raw Data 坐标验证")
    print("=" * 60)
    print(f"数据文件: {data_file}")
    print(f"图片目录: {image_root}")
    print(f"输出文件: {output_html}")
    print()
    
    # 检查文件是否存在
    if not os.path.exists(data_file):
        print(f"错误: 数据文件不存在: {data_file}")
        return
    
    if not os.path.exists(image_root):
        print(f"错误: 图片目录不存在: {image_root}")
        return
    
    # 加载数据
    print("正在加载数据...")
    data = load_jsonl(data_file)
    print(f"✓ 加载了 {len(data)} 条数据")
    
    # 随机抽取10个样本
    random.seed(42)
    if len(data) > 10:
        samples = random.sample(data, 10)
    else:
        samples = data
    print(f"✓ 抽取了 {len(samples)} 个样本进行验证")
    print()
    
    # 生成验证HTML
    print("正在生成验证HTML...")
    generate_verification_html(samples, image_root, output_html)
    print()
    print("=" * 60)
    print("✓ 验证完成！")
    print(f"请在浏览器中打开: {output_html}")
    print("=" * 60)


if __name__ == "__main__":
    main()

