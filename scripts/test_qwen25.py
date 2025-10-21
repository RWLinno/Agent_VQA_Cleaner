#!/usr/bin/env python
"""
Qwen2.5-VL-3B-Instruct 测试脚本
用于验证模型是否正常工作
"""

import os
import sys
from pathlib import Path

# 添加src目录到Python路径
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

def test_imports():
    """测试导入"""
    print("=" * 60)
    print("测试 1: 检查依赖导入")
    print("=" * 60)
    
    try:
        import torch
        print(f"✓ torch: {torch.__version__}")
    except ImportError as e:
        print(f"✗ torch 导入失败: {e}")
        return False
    
    try:
        import transformers
        print(f"✓ transformers: {transformers.__version__}")
    except ImportError as e:
        print(f"✗ transformers 导入失败: {e}")
        return False
    
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        print("✓ Qwen2.5-VL 模型类导入成功")
    except ImportError as e:
        print(f"✗ Qwen2.5-VL 模型类导入失败: {e}")
        print("\n请运行: pip install git+https://github.com/huggingface/transformers.git")
        return False
    
    try:
        from qwen_vl_utils import process_vision_info
        print("✓ qwen-vl-utils 导入成功")
    except ImportError:
        print("⚠ qwen-vl-utils 导入失败（可选）")
    
    try:
        import lmdeploy
        print(f"✓ lmdeploy: {lmdeploy.__version__}")
    except ImportError:
        print("⚠ lmdeploy 导入失败（可选，仅用于某些模型）")
    
    print()
    return True


def test_config():
    """测试配置"""
    print("=" * 60)
    print("测试 2: 检查配置")
    print("=" * 60)
    
    try:
        from src.config import Config
        
        print(f"✓ 配置加载成功")
        print(f"  - 主模型: {Config.PRIMARY_MODEL}")
        print(f"  - 备用模型1: {Config.SECONDARY_MODEL}")
        print(f"  - 并发处理器数量: {Config.NUM_PROCESSORS}")
        print(f"  - 批处理大小: {Config.BATCH_SIZE}")
        print(f"  - 评分阈值: {Config.SCORE_THRESHOLD}")
        print()
        return True
    except Exception as e:
        print(f"✗ 配置加载失败: {e}")
        print()
        return False


def test_model_simple():
    """简单模型测试（不实际加载模型）"""
    print("=" * 60)
    print("测试 3: 模型路径检查")
    print("=" * 60)
    
    try:
        from src.config import Config
        model_path, model_name = Config.get_available_model(auto_download=False)
        
        print(f"✓ 将使用模型: {model_name}")
        print(f"  - 路径: {model_path}")
        
        # 如果是HF模型ID（不是本地路径）
        if not os.path.exists(model_path) and model_path.startswith('Qwen/'):
            print(f"  - 类型: Hugging Face 模型ID（首次运行时将自动下载）")
        elif os.path.exists(model_path):
            print(f"  - 类型: 本地模型")
            print(f"  - 状态: ✓ 已存在")
        else:
            print(f"  - 状态: ⚠ 不存在（将在运行时下载）")
        
        print()
        return True
        
    except Exception as e:
        print(f"✗ 模型路径检查失败: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False


def test_model_load():
    """实际加载模型测试（可选）"""
    print("=" * 60)
    print("测试 4: 实际加载模型（可选，较慢）")
    print("=" * 60)
    
    response = input("是否要实际加载模型进行测试？这可能需要几分钟时间。(y/N): ")
    if response.lower() != 'y':
        print("⊙ 跳过模型加载测试")
        print()
        return True
    
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        from src.config import Config
        
        print("正在加载模型...")
        model_path = Config.PRIMARY_MODEL
        
        # 加载processor
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        print("✓ Processor 加载成功")
        
        # 加载模型（使用CPU或GPU）
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  - 使用设备: {device}")
        
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True
        )
        print("✓ 模型加载成功")
        
        # 简单推理测试
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "这是一个测试。"},
                ],
            }
        ]
        
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], return_tensors="pt")
        
        if device == "cuda":
            inputs = inputs.to("cuda")
        
        print("正在进行推理测试...")
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=10)
        
        output_text = processor.batch_decode(generated_ids, skip_special_tokens=True)
        print("✓ 推理测试成功")
        print(f"  - 输出: {output_text[0][:100]}...")
        
        print()
        return True
        
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        print()
        return False


def main():
    """主测试流程"""
    print("\n" + "=" * 60)
    print("Qwen2.5-VL-3B-Instruct 测试脚本")
    print("=" * 60)
    print()
    
    results = []
    
    # 测试1: 导入
    results.append(("导入检查", test_imports()))
    
    # 测试2: 配置
    results.append(("配置检查", test_config()))
    
    # 测试3: 模型路径
    results.append(("模型路径检查", test_model_simple()))
    
    # 测试4: 实际加载（可选）
    results.append(("模型加载测试", test_model_load()))
    
    # 总结
    print("=" * 60)
    print("测试总结")
    print("=" * 60)
    
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{status}: {name}")
    
    print()
    
    # 判断是否通过
    passed = sum(1 for _, r in results[:3] if r)  # 前3个测试必须通过
    if passed == 3:
        print("🎉 基础测试全部通过！系统已准备就绪。")
        print()
        print("下一步:")
        print("  - 查看安装指南: cat INSTALL_QWEN2.5.md")
        print("  - 运行快速开始: ./quick_start.sh")
        print("  - 或直接运行: python main.py --help")
        return 0
    else:
        print("⚠ 部分测试未通过，请检查安装。")
        print()
        print("安装步骤:")
        print("  1. 运行安装脚本: bash install_qwen25.sh")
        print("  2. 查看安装指南: cat INSTALL_QWEN2.5.md")
        return 1


if __name__ == "__main__":
    sys.exit(main())

