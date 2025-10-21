"""
AgentCleaner - 主程序入口
VQA数据清洗系统的主入口程序
"""
import os
import sys
import argparse
import logging
from pathlib import Path

src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.config import Config
from src.base import UnifiedManager


def setup_logging(log_level: str = "INFO", log_file: str = None):
    """
    设置日志系统
    
    Args:
        log_level: 日志级别
        log_file: 日志文件路径（可选）
    """
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # 基础配置
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    # 如果指定了日志文件，添加文件处理器
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="AgentCleaner - 基于多模态Agent的VQA数据清洗系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
  # 自定义配置
  python main.py --json_file data.json --root_path /path/to/images --output output.json \
      --num_processors 4 --batch_size 8 --score_threshold 6
        """
    )
    
    parser.add_argument(
        '--json_file',
        type=str,
        required=True,
        help='输入JSON文件路径'
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
        help='输出文件路径'
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
        '--include_answer',
        action='store_true',
        help='在评估时包含答案'
    )
    parser.add_argument(
        '--no_concurrent',
        action='store_true',
        help='禁用并发处理，使用顺序处理'
    )
    parser.add_argument(
        '--no_heuristic',
        action='store_true',
        help='禁用启发式规则过滤'
    )
    
    # 可选参数 - 配置覆盖
    parser.add_argument(
        '--num_processors',
        type=int,
        default=None,
        help=f'并发处理器数量 (默认: {Config.NUM_PROCESSORS})'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help=f'批处理大小 (默认: {Config.BATCH_SIZE})'
    )
    parser.add_argument(
        '--score_threshold',
        type=int,
        default=None,
        help=f'评分阈值 (默认: {Config.SCORE_THRESHOLD})'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default=None,
        help='自定义模型路径'
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
    
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logging(args.log_level, args.log_file)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 70)
    logger.info("AgentCleaner - VQA数据清洗系统启动")
    logger.info("=" * 70)
    
    if args.num_processors is not None:
        Config.NUM_PROCESSORS = args.num_processors
    if args.batch_size is not None:
        Config.BATCH_SIZE = args.batch_size
    if args.score_threshold is not None:
        Config.SCORE_THRESHOLD = args.score_threshold
    if args.model_path is not None:
        Config.PRIMARY_MODEL = args.model_path
    
    if not os.path.exists(args.json_file):
        logger.error(f"输入文件不存在: {args.json_file}")
        sys.exit(1)
    
    if not os.path.exists(args.root_path):
        logger.error(f"图像根目录不存在: {args.root_path}")
        sys.exit(1)
    
    try:
        manager = UnifiedManager(Config)
        
        manager.run(
            json_file=args.json_file,
            root_path=args.root_path,
            output_file=args.output,
            start=args.start,
            end=args.end,
            include_answer=args.include_answer,
            concurrent=not args.no_concurrent
        )
        
        logger.info("=" * 70)
        logger.info("数据清洗完成!")
        logger.info(f"清洗后的数据: {args.output}")
        full_output = args.output.replace('.jsonl', '_full.jsonl').replace('.json', '_full.jsonl')
        logger.info(f"完整结果（含评估信息）: {full_output}")
        prompt_output = args.output.replace('.jsonl', '_system_prompt.txt').replace('.json', '_system_prompt.txt')
        logger.info(f"System prompt: {prompt_output}")
        logger.info("=" * 70)
        
    except KeyboardInterrupt:
        logger.warning("\n用户中断程序")
        sys.exit(0)
    except Exception as e:
        logger.error(f"程序运行出错: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

