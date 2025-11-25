"""
点位验证主程序
用于处理洗衣机数据集的点位验证流程
"""
import os
import sys
import json
import argparse
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

# 添加src到路径
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

from src.config import Config
from src.base.vlm_agent_eas import VLMAgentEAS
from src.base.point_verification_processor import PointVerificationProcessor


def setup_logging(log_level: str = "INFO", log_file: str = None):
    """设置日志系统"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """加载JSONL文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict[str, Any]], output_path: str):
    """保存为JSONL文件"""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def process_batch(
    processor: PointVerificationProcessor,
    batch: List[Dict[str, Any]],
    root_path: str
) -> List[Dict[str, Any]]:
    """处理一个批次"""
    results = []
    for item in batch:
        try:
            result = processor.process_single(item, root_path)
            results.append(result)
        except Exception as e:
            logging.error(f"处理失败: {e}")
            results.append({
                'id': item.get('id', 'unknown'),
                'error': str(e),
                'status': 'error'
            })
    return results


def run_parallel(
    data: List[Dict[str, Any]],
    root_path: str,
    vlm_agent,
    config,
    num_workers: int = 4,
    distance_threshold: float = 50.0,
    enable_perturbation: bool = True,
    perturbation_range: int = 100
) -> List[Dict[str, Any]]:
    """并行处理数据"""
    logger = logging.getLogger(__name__)
    logger.info(f"开始并行处理 {len(data)} 条数据，使用 {num_workers} 个worker")
    logger.info(f"配置: 距离阈值={distance_threshold}px, 扰动验证={enable_perturbation}, 扰动范围={perturbation_range}px")
    
    # 创建处理器
    processors = []
    for i in range(num_workers):
        processor = PointVerificationProcessor(
            processor_id=i,
            vlm_agent=vlm_agent,
            config=config,
            distance_threshold=distance_threshold,
            enable_perturbation=enable_perturbation,
            perturbation_range=perturbation_range
        )
        processors.append(processor)
    
    # 分割数据
    chunk_size = len(data) // num_workers
    remainder = len(data) % num_workers
    
    chunks = []
    start_idx = 0
    for i in range(num_workers):
        end_idx = start_idx + chunk_size + (1 if i < remainder else 0)
        chunks.append(data[start_idx:end_idx])
        start_idx = end_idx
    
    # 并行处理
    all_results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for processor, chunk in zip(processors, chunks):
            if len(chunk) > 0:
                future = executor.submit(process_batch, processor, chunk, root_path)
                futures.append(future)
        
        for future in as_completed(futures):
            try:
                results = future.result()
                all_results.extend(results)
                logger.info(f"已完成 {len(all_results)}/{len(data)}")
            except Exception as e:
                logger.error(f"收集结果时出错: {e}")
    
    # 打印统计信息
    logger.info("\n" + "=" * 70)
    logger.info("各Worker处理统计")
    logger.info("=" * 70)
    for processor in processors:
        stats = processor.get_statistics()
        logger.info(f"\n📊 Processor {stats['processor_id']}:")
        logger.info(f"  总处理: {stats['processed']} 条")
        logger.info(
            f"  点位匹配: {stats['point_comparison']['match']}/{stats['processed']} "
            f"({stats['point_comparison']['match_rate']:.1f}%)"
        )
        logger.info(
            f"  PointQA正确: {stats['pointqa_verification']['correct']}/{stats['processed']} "
            f"({stats['pointqa_verification']['correct_rate']:.1f}%)"
        )
        if enable_perturbation:
            logger.info(
                f"  负样本检测: {stats['negative_sample']['detected']}/{stats['processed']} "
                f"({stats['negative_sample']['detect_rate']:.1f}%)"
            )
        logger.info(
            f"  严格筛选通过: {stats['strict_filtering']['pass']}/{stats['processed']} "
            f"({stats['strict_filtering']['pass_rate']:.1f}%)"
        )
    logger.info("=" * 70 + "\n")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="点位验证主程序 - 处理洗衣机数据集的点位验证",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='输入JSONL文件路径'
    )
    parser.add_argument(
        '--root_path',
        type=str,
        required=True,
        help='图像根目录路径'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='输出JSONL文件路径'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='并行worker数量 (默认: 4)'
    )
    parser.add_argument(
        '--distance_threshold',
        type=float,
        default=50.0,
        help='点位距离阈值（像素） (默认: 50)'
    )
    parser.add_argument(
        '--enable_perturbation',
        type=lambda x: x.lower() == 'true',
        default=True,
        help='是否启用点位扰动验证 (默认: True)'
    )
    parser.add_argument(
        '--perturbation_range',
        type=int,
        default=100,
        help='点位扰动范围（像素） (默认: 100)'
    )
    parser.add_argument(
        '--start',
        type=int,
        default=0,
        help='起始索引 (默认: 0)'
    )
    parser.add_argument(
        '--end',
        type=int,
        default=None,
        help='结束索引 (默认: None，处理到最后)'
    )
    parser.add_argument(
        '--log_level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='日志级别 (默认: INFO)'
    )
    parser.add_argument(
        '--log_file',
        type=str,
        default=None,
        help='日志文件路径（可选）'
    )
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 70)
    logger.info("点位验证系统启动")
    logger.info("=" * 70)
    
    # 设置环境变量
    os.environ['POINT_DISTANCE_THRESHOLD'] = str(args.distance_threshold)
    
    # 打印配置
    Config.print_config()
    
    # 检查输入文件
    if not os.path.exists(args.input):
        logger.error(f"输入文件不存在: {args.input}")
        sys.exit(1)
    
    if not os.path.exists(args.root_path):
        logger.error(f"图像根目录不存在: {args.root_path}")
        sys.exit(1)
    
    try:
        # 初始化VLM Agent
        logger.info("正在初始化VLM Agent...")
        if Config.AGENT_TYPE == 'eas':
            vlm_agent = VLMAgentEAS(
                base_url=Config.EAS_BASE_URL,
                token=Config.EAS_TOKEN,
                model_name=Config.EAS_MODEL_NAME,
                max_tokens=Config.EAS_MAX_TOKENS,
                timeout=Config.EAS_TIMEOUT
            )
            logger.info(f"使用阿里云EAS API: {Config.EAS_MODEL_NAME}")
        else:
            logger.error("当前仅支持EAS模式，请设置 AGENT_TYPE=eas")
            sys.exit(1)
        
        # 加载数据
        logger.info(f"正在加载数据: {args.input}")
        data = load_jsonl(args.input)
        logger.info(f"✓ 加载了 {len(data)} 条数据")
        
        # 切片数据
        if args.end is not None:
            data = data[args.start:args.end]
        else:
            data = data[args.start:]
        logger.info(f"处理数据范围: {args.start} 到 {args.start + len(data)}")
        
        # 并行处理
        results = run_parallel(
            data=data,
            root_path=args.root_path,
            vlm_agent=vlm_agent,
            config=Config,
            num_workers=args.num_workers,
            distance_threshold=args.distance_threshold,
            enable_perturbation=args.enable_perturbation,
            perturbation_range=args.perturbation_range
        )
        
        # 保存结果
        logger.info(f"正在保存结果到: {args.output}")
        save_jsonl(results, args.output)
        logger.info(f"✓ 结果已保存: {args.output}")
        
        # 统计信息
        total = len(results)
        matched = sum(1 for r in results if r.get('point_comparison', {}).get('match', False))
        pointqa_correct = sum(1 for r in results if r.get('pointqa_verification', {}).get('is_correct', False))
        negative_detected = sum(1 for r in results if r.get('perturbation_verification', {}).get('negative_detected', False))
        strict_pass = sum(1 for r in results if r.get('strict_filtering', {}).get('pass', False))
        errors = sum(1 for r in results if r.get('status') == 'error')
        
        logger.info("\n" + "=" * 70)
        logger.info("📊 最终统计汇总")
        logger.info("=" * 70)
        logger.info(f"总处理数据: {total} 条")
        logger.info(f"")
        logger.info(f"🎯 点位匹配: {matched}/{total} ({matched/total*100:.1f}%)")
        logger.info(f"   - P1和P2距离 < {args.distance_threshold}px")
        logger.info(f"")
        logger.info(f"✅ PointQA正确: {pointqa_correct}/{total} ({pointqa_correct/total*100:.1f}%)")
        logger.info(f"   - 原始点P1指向正确位置")
        logger.info(f"")
        if args.enable_perturbation:
            logger.info(f"🔍 负样本检测: {negative_detected}/{total} ({negative_detected/total*100:.1f}%)")
            logger.info(f"   - 能正确识别扰动点P1'为错误")
            logger.info(f"")
        logger.info(f"⭐ 严格筛选通过: {strict_pass}/{total} ({strict_pass/total*100:.1f}%)")
        logger.info(f"   - 同时满足所有验证条件")
        logger.info(f"")
        logger.info(f"❌ 处理错误: {errors}")
        logger.info("=" * 70)
        
        logger.info("\n✓ 点位验证完成！")
        logger.info(f"\n下一步：使用可视化工具查看结果")
        logger.info(f"  python visualize_point_verification.py \\")
        logger.info(f"    --input {args.output} \\")
        logger.info(f"    --output visualization.html \\")
        logger.info(f"    --filter all")
        
    except KeyboardInterrupt:
        logger.warning("\n用户中断程序")
        sys.exit(0)
    except Exception as e:
        logger.error(f"程序运行出错: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

