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
    
    # 主模型: Qwen3-VL-30B-A3B-Thinking (本地模型)
    PRIMARY_MODEL = os.environ.get(
        'PRIMARY_MODEL', 
        '/mnt/data/huggingface_downloads/models/qwen/Qwen3-VL-30B-A3B-Thinking'
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
You are an expert annotator specializing in spatial reasoning and visual question answering evaluation. Please evaluate the image and question together, providing a score from 0 to 10 based on the detailed criteria below:

【Scoring Guidelines】
10 = Perfect: Exceptional image quality with perfect object visibility, clarity, and unambiguous question.
9 = Excellent: High-quality image with clear objects, well-formulated question with minor imperfections.
8 = Very Good: Good image quality, objects are clearly visible, question is clear with acceptable phrasing.
7 = Good: Acceptable image quality, objects identifiable but with minor clarity issues, question is understandable.
6 = Satisfactory: Image has noticeable quality issues but objects are still recognizable, question is somewhat clear.
5 = Borderline: Image quality is mediocre or question has ambiguity that affects interpretation.
4 = Poor: Image has significant quality issues or question is unclear, affecting usability.
3 = Very Poor: Major defects in image or question that severely limit usability.
2 = Severely Defective: Image or question is barely usable with critical flaws.
1 = Unusable: Almost completely unusable due to severe image/question defects.
0 = Completely Unusable: Impossible to use for annotation or training.

【Evaluation Criteria】

Image Quality (50% weight):
- Realism & Clarity: Is the image photorealistic? Are textures, lighting, and shadows natural?
- Object Definition: Are object boundaries clear and well-defined? Can you distinguish object edges easily?
- Visibility: Are all objects mentioned in the question fully visible without significant occlusion?
- Color Accuracy: Do objects have realistic colors appropriate to their type?
- Resolution & Focus: Is the image sharp and in focus? Are there blur or pixelation issues?
- Lighting: Is lighting appropriate and consistent? Does it reveal object details adequately?

Question Quality (50% weight):
- Clarity: Is the question grammatically correct and easy to understand?
- Specificity: Does the question specify objects/parts clearly without ambiguity?
- Relevance: Can the question be answered using information visible in the image?
- Spatial Reasoning: For spatial questions, are spatial relationships clearly queryable from the image?
- Completeness: Does the question contain all necessary context to be answered?

【Critical Failure Scenarios - Force Score ≤ 3】
- Object in question is not present or completely occluded in the image
- Objects have severe unrealistic appearance (unnatural shapes, impossible colors, distorted geometry)
- Critical parts referenced in the question are invisible or extremely unclear
- Question contains ambiguous pronouns or unclear referents that cannot be resolved from image
- Image has severe artifacts, corruption, or rendering errors
- Spatial relationships mentioned in question are impossible to determine from the image
- Question contains logical contradictions or is unanswerable by design

【Special Bonus Conditions - May increase score by +1】
- Multiple objects are clearly distinguishable with excellent contrast
- Question tests complex spatial reasoning with clear visual support
- Image includes useful depth cues (shadows, perspective) for spatial understanding

Please analyze the image and question carefully, then provide ONLY a single number from 0 to 10 as your final score.

The question is:
"""
    
    # 系统提示词 - 包含答案的评估
    SYSTEM_PROMPT_WITH_ANSWER = """
You are an expert annotator specializing in visual question answering evaluation. Please evaluate the image, question, and answer as a complete triplet, providing a score from 0 to 10 based on the detailed criteria below:

【Scoring Guidelines】
10 = Perfect: Exceptional quality across all three components - image is flawless, question is precise, answer is completely accurate and well-articulated.
9 = Excellent: All three components are high quality with only trivial imperfections.
8 = Very Good: Strong quality overall, minor issues in one component that don't significantly affect usability.
7 = Good: Generally good quality, some noticeable issues but the triplet remains useful for training.
6 = Satisfactory: Acceptable quality with recognizable issues in one or more components, but still usable.
5 = Borderline: Mediocre quality that is barely acceptable, significant issues that affect training value.
4 = Poor: Significant quality issues in one or more components that reduce usability.
3 = Very Poor: Major defects that severely limit the training value of this sample.
2 = Severely Defective: Multiple critical flaws making the sample barely usable.
1 = Unusable: Extremely poor quality, should not be used for training.
0 = Completely Unusable: Cannot be used under any circumstances.

【Evaluation Criteria】

Image Quality (30% weight):
- Realism: Is the image photorealistic with natural appearance?
- Clarity: Are objects sharp, well-defined, and easily distinguishable?
- Visibility: Are all relevant objects/regions fully visible and unoccluded?
- Technical Quality: No severe artifacts, appropriate resolution and lighting?

Question Quality (30% weight):
- Clarity: Is the question grammatically correct and unambiguous?
- Specificity: Does it clearly specify what is being asked?
- Answerability: Can the question be answered from the visual information provided?
- Relevance: Is the question appropriate for spatial reasoning or VQA tasks?

Answer Quality (40% weight):
- Factual Accuracy: Is the answer correct based on the image content?
- Completeness: Does the answer fully address the question without omissions?
- Coherence: Is the answer logically structured and easy to understand?
- Conciseness: Is the answer appropriately detailed without unnecessary verbosity?
- Format Quality: Is the answer properly formatted without repetition, loops, or truncation?
- Linguistic Quality: Is the grammar correct and the language natural?

【Critical Failure Scenarios - Force Score ≤ 3】
Answer Accuracy Issues:
- Answer is factually incorrect or contradicts visible image content
- Answer describes objects or spatial relationships not present in the image
- Answer provides information that cannot be verified from the image

Answer Format Issues:
- Contains obvious repetitive patterns (e.g., repeated phrases, loops)
- Has unnatural text repetition in the tail (last 20+ characters repeated 5+ times)
- Is incomplete, truncated, or cuts off mid-sentence
- Contains obvious generation artifacts or malformed text

Coherence Issues:
- Answer is logically inconsistent or self-contradictory
- Answer does not actually address the question asked
- Answer switches topics or loses focus mid-response

Image-Question-Answer Mismatch:
- Question cannot be answered from the image provided
- Answer references information not queryable from the image
- Severe ambiguity that prevents determining answer correctness

【Special Bonus Conditions - May increase score by +1】
- Answer provides insightful spatial reasoning with clear justification
- Answer demonstrates excellent understanding of complex spatial relationships
- Question-answer pair is particularly well-suited for training spatial reasoning models
- Answer includes appropriate level of detail that enhances learning value

【Automatic Penalties】
- Repetitive n-grams (40%+ repetition): -3 points
- Extremely long sentences (50+ meaningful words): -2 points
- Tail repetition loops: -3 points
- Truncated or incomplete answer: -2 points

Please analyze all three components carefully and holistically, then provide ONLY a single number from 0 to 10 as your final score.

The question and answer are:
"""
    
    # 启发式规则配置 - 更细致的规则参数
    HEURISTIC_CONFIG = {
        # N-gram重复检测
        'ngram': 10,
        'repeat_threshold': 0.35,
        
        # 超长句子检测
        'super_long_sentence_words': 25, 
        'extreme_long_sentence_words': 45,  
        
        # 尾部重复检测
        'tail_repeat_length': 15, 
        'tail_repeat_count': 5,  
        
        # 新增：短答案重复检测
        'short_answer_ngram': 5,  # 对短答案用5-gram检测
        'short_answer_threshold': 0.5,  # 短答案的重复阈值50%
        
        # 新增：词汇多样性检测
        'min_unique_word_ratio': 0.3, 
        
        # 新增：异常字符检测
        'max_special_char_ratio': 0.1,  # 特殊字符不应超过10%
        'suspicious_patterns': [ 
            '0000000000',
            '1111111111', 
            'aaaaaaaaaa',
            '...........',
            '----------',
        ]
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
        
        # 直接使用 PRIMARY_MODEL (Qwen3-VL-30B-A3B-Thinking)
        model_name = 'Qwen3-VL-30B-A3B-Thinking'
        model_path = cls.PRIMARY_MODEL
        
        logger.info(f"使用模型: {model_name}")
        logger.info(f"模型路径: {model_path}")
        
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

