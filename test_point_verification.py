"""
点位验证功能测试脚本
用于快速测试核心功能
"""
import os
import sys
import json
from pathlib import Path

# 添加src到路径
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

from src.config import Config
from src.base.vlm_agent_eas import VLMAgentEAS
from src.base.point_verification_processor import PointVerificationProcessor


def test_label_extraction():
    """测试label提取功能"""
    print("\n" + "=" * 60)
    print("测试1: Label提取功能")
    print("=" * 60)
    
    processor = PointVerificationProcessor(
        processor_id=0,
        vlm_agent=None,
        config=Config
    )
    
    test_cases = [
        'Point to all occurrences of "power button".',
        'Locate a start/pause button.',
        'If there are any control panel in the image?',
        'Find the door handle.',
        'Show me where the control knob is.'
    ]
    
    for question in test_cases:
        label = processor.extract_label_from_question(question)
        print(f"Question: {question}")
        print(f"  -> Label: {label}\n")
    
    print("✓ Label提取测试完成\n")


def test_point_extraction():
    """测试点坐标提取功能"""
    print("\n" + "=" * 60)
    print("测试2: 点坐标提取功能")
    print("=" * 60)
    
    processor = PointVerificationProcessor(
        processor_id=0,
        vlm_agent=None,
        config=Config
    )
    
    test_answer = """```json
[
\t{"point_2d": [420.91, 263.16], "label": "power button"}
]
```"""
    
    points = processor.extract_points_from_answer(test_answer)
    print(f"Answer: {test_answer}")
    print(f"  -> 提取的点: {points}\n")
    
    print("✓ 点坐标提取测试完成\n")


def test_distance_calculation():
    """测试距离计算功能"""
    print("\n" + "=" * 60)
    print("测试3: 距离计算功能")
    print("=" * 60)
    
    processor = PointVerificationProcessor(
        processor_id=0,
        vlm_agent=None,
        config=Config,
        distance_threshold=50.0
    )
    
    p1 = (420.91, 263.16)
    p2_near = (425.0, 265.0)  # 接近的点
    p2_far = (500.0, 350.0)   # 远离的点
    
    dist_near = processor.calculate_distance(p1, p2_near)
    dist_far = processor.calculate_distance(p1, p2_far)
    
    print(f"P1: {p1}")
    print(f"P2 (near): {p2_near}")
    print(f"  -> 距离: {dist_near:.2f}px")
    print(f"  -> 匹配: {dist_near <= processor.distance_threshold}\n")
    
    print(f"P1: {p1}")
    print(f"P2 (far): {p2_far}")
    print(f"  -> 距离: {dist_far:.2f}px")
    print(f"  -> 匹配: {dist_far <= processor.distance_threshold}\n")
    
    print("✓ 距离计算测试完成\n")


def test_eas_connection():
    """测试EAS API连接"""
    print("\n" + "=" * 60)
    print("测试4: EAS API连接测试")
    print("=" * 60)
    
    try:
        # 设置环境变量
        os.environ['AGENT_TYPE'] = 'eas'
        
        vlm_agent = VLMAgentEAS(
            base_url=Config.EAS_BASE_URL,
            token=Config.EAS_TOKEN,
            model_name=Config.EAS_MODEL_NAME,
            max_tokens=64,  # 测试用较小的max_tokens
            timeout=30
        )
        
        print(f"✓ EAS Agent初始化成功")
        print(f"  - 模型: {Config.EAS_MODEL_NAME}")
        print(f"  - URL: {Config.EAS_BASE_URL[:50]}...\n")
        
        # 测试简单推理
        print("测试简单文本推理...")
        response = vlm_agent.inference_single(
            prompt_text="请回答：1+1等于几？",
            image_path=None
        )
        print(f"模型响应: {response[:100]}\n")
        
        print("✓ EAS API连接测试完成\n")
        return vlm_agent
        
    except Exception as e:
        print(f"✗ EAS API连接失败: {e}\n")
        return None


def test_single_sample_processing(vlm_agent):
    """测试单个样本处理"""
    print("\n" + "=" * 60)
    print("测试5: 单个样本处理")
    print("=" * 60)
    
    if vlm_agent is None:
        print("跳过（需要有效的VLM Agent）\n")
        return
    
    # 加载一个测试样本
    data_file = "/mnt/data/datasets/knowin_datasets/washing_machine_1029/processed_washing_machine_data_all/train_single_label_float.jsonl"
    
    if not os.path.exists(data_file):
        print(f"数据文件不存在: {data_file}\n")
        return
    
    # 读取第一条数据
    with open(data_file, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
        if not first_line:
            print("数据文件为空\n")
            return
        
        sample = json.loads(first_line)
    
    print(f"样本ID: {sample.get('id', 'unknown')}")
    print(f"问题: {sample['messages'][0]['content'][:50]}...")
    
    # 创建处理器
    processor = PointVerificationProcessor(
        processor_id=0,
        vlm_agent=vlm_agent,
        config=Config,
        distance_threshold=50.0
    )
    
    # 处理样本
    root_path = "/mnt/data/datasets/knowin_datasets/washing_machine_1029/processed_washing_machine_data_all"
    
    try:
        print("\n开始处理...")
        result = processor.process_single(sample, root_path)
        
        print("\n处理结果:")
        print(f"  - Label: {result.get('label', 'N/A')}")
        print(f"  - P1 (原始): {result.get('point_original', {}).get('coordinates', 'N/A')}")
        print(f"  - P2 (推理): {result.get('point_inferred', {}).get('coordinates', 'N/A')}")
        
        comparison = result.get('point_comparison', {})
        print(f"  - 距离: {comparison.get('distance', 'N/A'):.2f}px" if comparison.get('distance') else "  - 距离: N/A")
        print(f"  - 匹配: {comparison.get('match', False)}")
        
        verification = result.get('pointqa_verification', {})
        print(f"  - PointQA结果: {verification.get('is_correct', False)}")
        print(f"  - 置信度: {verification.get('confidence', 0):.1f}/10")
        
        print("\n✓ 单个样本处理测试完成\n")
        
        return result
        
    except Exception as e:
        print(f"\n✗ 处理失败: {e}\n")
        import traceback
        traceback.print_exc()
        return None


def main():
    """主测试函数"""
    print("=" * 60)
    print("点位验证功能测试")
    print("=" * 60)
    
    # 测试1: Label提取
    test_label_extraction()
    
    # 测试2: 点坐标提取
    test_point_extraction()
    
    # 测试3: 距离计算
    test_distance_calculation()
    
    # 测试4: EAS连接
    vlm_agent = test_eas_connection()
    
    # 测试5: 单个样本处理（需要EAS连接成功）
    if vlm_agent:
        result = test_single_sample_processing(vlm_agent)
        
        if result:
            # 保存测试结果
            output_file = "/mnt/users/rwl/Agent_VQA_Cleaner/output/test_result.json"
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"测试结果已保存到: {output_file}")
    
    print("\n" + "=" * 60)
    print("所有测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
