"""
配置文件 - 定义模型路径、并发参数等配置
"""
import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

class Config:
    """全局配置类"""
    
    # 模型配置
    # 优先使用环境变量，否则使用预设路径，最后使用本地models目录
    
    # 主模型: Qwen2.5-VL-3B-Instruct (直接从HF下载)
    # 使用 HF 模型 ID，避免本地模型文件损坏问题
    PRIMARY_MODEL = os.environ.get(
        'PRIMARY_MODEL', 
        'Qwen/Qwen2.5-VL-3B-Instruct'  # 直接使用HF模型ID，会自动下载
    )
    
    SECONDARY_MODEL = os.environ.get(
        'SECONDARY_MODEL',
        str(MODELS_DIR / 'Qwen' / 'Qwen3-VL-8B-Thinking')
    )
    
    INTERNVL_MODEL = os.environ.get(
        'INTERNVL_MODEL',
        str(MODELS_DIR / 'InternVL3.5-38B')
    )
    
    # Hugging Face模型ID映射
    HF_MODEL_MAP = {
        'Qwen2.5-VL-3B-Instruct': 'Qwen/Qwen2.5-VL-3B-Instruct',  # ⭐ 最优选：小而强
        'Qwen3-VL-8B-Thinking': 'Qwen/Qwen3-VL-8B-Thinking',
        'InternVL3.5-38B': 'OpenGVLab/InternVL2_5-38B',
        'Qwen3-VL-30B': 'Qwen/Qwen2-VL-7B-Instruct',
    }
    
    # 自动下载配置
    AUTO_DOWNLOAD = os.environ.get('AUTO_DOWNLOAD', 'true').lower() == 'true'
    MODELS_CACHE_DIR = str(MODELS_DIR)
    
    # 并发配置
    NUM_PROCESSORS = int(os.environ.get('NUM_PROCESSORS', 8))  # 默认8个并发processor
    BATCH_SIZE = int(os.environ.get('BATCH_SIZE', 4))  # 每个processor的批处理大小
    
    # GPU配置
    CUDA_VISIBLE_DEVICES = os.environ.get('CUDA_VISIBLE_DEVICES', '0,1,2,3')
    TP = len(CUDA_VISIBLE_DEVICES.split(',')) if CUDA_VISIBLE_DEVICES else 1
    
    # 模型推理配置
    SESSION_LEN = 16000
    MAX_RETRIES = 3  # 失败重试次数
    
    # 评分阈值
    SCORE_THRESHOLD = 5  # 低于此分数的数据将被过滤
    
    # 系统提示词 - 图像质量+问题质量评估
    SYSTEM_PROMPT_IMAGE_QUESTION = """
You are an expert annotator. Evaluate the image and the question (related to spatial reasoning or visual question answering) together and give a single overall score from 0 to 10, where:
10 = Excellent: Image is clear and realistic, objects are well-defined, and the question is precise and unambiguous.
0 = Unusable: Image or question is too poor to allow reliable annotation.

Consider the following criteria when forming your judgment:

Image Quality:
- Is the image realistic and clear?
- Is the lighting appropriate?
- Are the objects in the image well-defined and visible?
- Are the objects in the image realistic in terms of shape and color?
- Is the object asked in the question well-defined, visible and with clear boundary?

Question Quality:
- Can the object specified in the question be clearly identified in the image?
- Is the question clear and unambiguous?
- If the question is about a part of the object, is that part clearly visible in the image?

Scoring Logic:
10 is the best, 0 is the worst.

Critical Failures (force score ≤ 3):
- Objects look unrealistic or have unnatural colors.
- The object mentioned in the question cannot be clearly identified.
- The relevant object part is completely occluded.
- The question is ambiguous or unclear.

Please only provide the final score, only give me one number.
The question is:
"""
    
    # 系统提示词 - 包含答案的评估
    SYSTEM_PROMPT_WITH_ANSWER = """
You are an expert annotator. Evaluate the image, question, and answer together and give a single overall score from 0 to 10, where:
10 = Excellent: Image is clear, question is precise, and the answer is accurate and well-formatted.
0 = Unusable: Image, question, or answer is too poor to be useful.

Consider the following criteria:

Image Quality:
- Is the image realistic and clear?
- Are objects well-defined and visible?

Question Quality:
- Is the question clear and unambiguous?
- Can it be answered based on the image?

Answer Quality:
- Is the answer accurate based on the image?
- Is the answer properly formatted and complete?
- Does the answer contain repetitive or looping text?
- Does the answer make logical sense?

Scoring Logic:
10 is the best, 0 is the worst.

Critical Failures (force score ≤ 3):
- The answer is factually incorrect.
- The answer contains repetitive patterns or loops.
- The answer is incomplete or truncated.
- The question cannot be answered from the image.

Please only provide the final score, only give me one number.
The question and answer are:
"""
    
    # 启发式规则配置
    HEURISTIC_CONFIG = {
        'ngram': 10,
        'repeat_threshold': 0.4,
        'super_long_sentence_words': 20,
        'extreme_long_sentence_words': 50,
        'tail_repeat_length': 20,
        'tail_repeat_count': 8,
    }
    
    # 输出配置
    SAVE_INTERVAL = 100  # 每处理多少条数据保存一次中间结果
    OUTPUT_FORMAT = 'jsonl'  # 输出格式: json 或 jsonl
    
    @classmethod
    def download_model_from_hf(cls, model_name: str, local_dir: str = None):
        """
        从Hugging Face下载模型
        
        Args:
            model_name: 模型名称（如'InternVL3.5-38B'）
            local_dir: 本地保存目录，默认为./models/model_name
            
        Returns:
            下载后的模型路径
        """
        try:
            from huggingface_hub import snapshot_download
            import logging
            logger = logging.getLogger(__name__)
            
            # 获取HF模型ID
            hf_model_id = cls.HF_MODEL_MAP.get(model_name)
            if not hf_model_id:
                raise ValueError(f"未找到模型 {model_name} 的Hugging Face映射")
            
            # 设置本地目录
            if local_dir is None:
                local_dir = os.path.join(cls.MODELS_CACHE_DIR, model_name)
            
            os.makedirs(local_dir, exist_ok=True)
            
            logger.info(f"正在从Hugging Face下载模型: {hf_model_id}")
            logger.info(f"保存到: {local_dir}")
            logger.info("这可能需要一些时间，请耐心等待...")
            
            # 下载模型
            model_path = snapshot_download(
                repo_id=hf_model_id,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
                resume_download=True
            )
            
            logger.info(f"模型下载完成: {model_path}")
            return model_path
            
        except ImportError:
            raise ImportError(
                "需要安装 huggingface_hub 来下载模型。\n"
                "请运行: pip install huggingface_hub"
            )
        except Exception as e:
            raise RuntimeError(f"模型下载失败: {e}")
    
    @classmethod
    def get_available_model(cls, auto_download: bool = None):
        """
        获取可用的模型路径，始终优先返回 PRIMARY_MODEL
        
        Args:
            auto_download: 是否自动下载，默认使用配置中的AUTO_DOWNLOAD
            
        Returns:
            (model_path, model_name)
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # 直接使用 PRIMARY_MODEL (Qwen2.5-VL-3B-Instruct)
        # 这是 HF 模型 ID，transformers 会自动下载
        model_name = 'Qwen2.5-VL-3B-Instruct'
        model_path = cls.PRIMARY_MODEL
        
        logger.info(f"使用默认模型: {model_name}")
        logger.info(f"模型路径/ID: {model_path}")
        
        return model_path, model_name
    
    @classmethod
    def print_config(cls):
        """打印当前配置"""
        model_path, model_name = cls.get_available_model()
        print("=" * 60)
        print("AgentCleaner 配置信息")
        print("=" * 60)
        print(f"使用模型: {model_name} ⭐")
        print(f"模型路径: {model_path}")
        print(f"并发处理器数量: {cls.NUM_PROCESSORS}")
        print(f"批处理大小: {cls.BATCH_SIZE}")
        print(f"GPU设置: {cls.CUDA_VISIBLE_DEVICES}")
        print(f"Tensor Parallel: {cls.TP}")
        print(f"评分阈值: {cls.SCORE_THRESHOLD}")
        print(f"保存间隔: {cls.SAVE_INTERVAL}")
        print(f"输出格式: {cls.OUTPUT_FORMAT}")
        print("=" * 60)

