#!/usr/bin/env python
"""
测试阿里云EAS API连接
用于验证EAS服务配置是否正确
"""
import os
import sys
from pathlib import Path

# 添加src到路径
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

from src.base.vlm_agent_eas import VLMAgentEAS
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_text_only():
    """测试纯文本推理（无图像）"""
    logger.info("=" * 60)
    logger.info("测试1: 纯文本推理")
    logger.info("=" * 60)
    
    # 从环境变量获取配置
    base_url = os.environ.get(
        'EAS_BASE_URL',
        'http://1054059136692489.cn-shanghai.pai-eas.aliyuncs.com/api/predict/quickstart_deploy_20251021_p4ar'
    )
    token = os.environ.get(
        'EAS_TOKEN',
        'ZmU3OTQxMGVlODFmYzZiNjQ3MTBlOWQ3NGMzZGQ4NzFlNmNkZjZlYQ=='
    )
    model_name = os.environ.get('EAS_MODEL_NAME', 'Qwen3-VL-235B-A22B-Instruct')
    
    try:
        # 初始化Agent
        agent = VLMAgentEAS(
            base_url=base_url,
            token=token,
            model_name=model_name,
            max_tokens=100,
            timeout=30
        )
        
        # 测试简单问题
        prompt = "请用一句话介绍你自己。"
        logger.info(f"发送提示: {prompt}")
        
        response = agent.inference_single(prompt)
        
        logger.info(f"收到响应: {response}")
        
        if "Error" not in response:
            logger.info("✅ 测试通过：EAS API连接正常")
            return True
        else:
            logger.error(f"❌ 测试失败：{response}")
            return False
            
    except Exception as e:
        logger.error(f"❌ 测试失败：{e}")
        return False


def test_with_image():
    """测试带图像的推理"""
    logger.info("\n" + "=" * 60)
    logger.info("测试2: 图像推理（需要提供测试图像）")
    logger.info("=" * 60)
    
    # 检查测试图像
    test_image = Path(__file__).parent / "examples" / "images"
    if test_image.exists():
        image_files = list(test_image.glob("*.jpg")) + list(test_image.glob("*.png"))
        if image_files:
            test_image_path = str(image_files[0])
            logger.info(f"找到测试图像: {test_image_path}")
        else:
            logger.warning("未找到测试图像，跳过图像推理测试")
            return None
    else:
        logger.warning("examples/images 目录不存在，跳过图像推理测试")
        return None
    
    base_url = os.environ.get(
        'EAS_BASE_URL',
        'http://1054059136692489.cn-shanghai.pai-eas.aliyuncs.com/api/predict/quickstart_deploy_20251021_p4ar'
    )
    token = os.environ.get(
        'EAS_TOKEN',
        'ZmU3OTQxMGVlODFmYzZiNjQ3MTBlOWQ3NGMzZGQ4NzFlNmNkZjZlYQ=='
    )
    model_name = os.environ.get('EAS_MODEL_NAME', 'Qwen3-VL-235B-A22B-Instruct')
    
    try:
        agent = VLMAgentEAS(
            base_url=base_url,
            token=token,
            model_name=model_name,
            max_tokens=100,
            timeout=60
        )
        
        prompt = "请描述这张图像的内容。"
        logger.info(f"发送提示: {prompt}")
        logger.info(f"图像: {test_image_path}")
        
        response = agent.inference_single(prompt, image_path=test_image_path)
        
        logger.info(f"收到响应: {response}")
        
        if "Error" not in response:
            logger.info("✅ 测试通过：图像推理正常")
            return True
        else:
            logger.error(f"❌ 测试失败：{response}")
            return False
            
    except Exception as e:
        logger.error(f"❌ 测试失败：{e}")
        return False


def main():
    """主测试函数"""
    logger.info("\n" + "🔍 开始测试阿里云EAS API连接")
    logger.info("=" * 60)
    
    # 显示配置
    logger.info("当前配置:")
    logger.info(f"  EAS_BASE_URL: {os.environ.get('EAS_BASE_URL', '使用默认值')}")
    logger.info(f"  EAS_TOKEN: {'已设置' if os.environ.get('EAS_TOKEN') else '使用默认值'}")
    logger.info(f"  EAS_MODEL_NAME: {os.environ.get('EAS_MODEL_NAME', '使用默认值')}")
    logger.info("=" * 60 + "\n")
    
    results = []
    
    # 测试1：纯文本
    result1 = test_text_only()
    results.append(("纯文本推理", result1))
    
    # 测试2：图像推理
    result2 = test_with_image()
    if result2 is not None:
        results.append(("图像推理", result2))
    
    # 总结
    logger.info("\n" + "=" * 60)
    logger.info("测试总结")
    logger.info("=" * 60)
    for test_name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        logger.info(f"{test_name}: {status}")
    
    all_passed = all(r for _, r in results if r is not None)
    
    if all_passed:
        logger.info("\n🎉 所有测试通过！EAS API配置正确，可以开始使用。")
        return 0
    else:
        logger.info("\n⚠️  部分测试失败，请检查EAS配置和网络连接。")
        return 1


if __name__ == "__main__":
    sys.exit(main())

