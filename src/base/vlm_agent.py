"""
VLM Agent - 多模态大模型代理
负责加载和调用视觉语言模型进行推理
支持两种加载方式：
1. transformers 直接加载（用于 Qwen2.5-VL 等新模型）
2. lmdeploy 加载（用于其他模型）
"""
import os
import torch
from typing import List, Tuple, Dict, Any, Optional
import logging
from PIL import Image

logger = logging.getLogger(__name__)


class VLMAgent:

    def __init__(self, model_path: str, tp: int = 1, session_len: int = 16000):
        """
        初始化VLM Agent
        
        Args:
            model_path: 模型路径
            tp: Tensor Parallel数量（根据GPU数量设置）
            session_len: 会话长度
        """
        self.model_path = model_path
        self.tp = tp
        self.session_len = session_len
        self.pipe = None
        self.model = None
        self.processor = None
        self.use_transformers = False  # 标记是否使用transformers直接加载
        self.model_name = os.path.basename(model_path)
        
        logger.info(f"初始化VLM Agent: {self.model_name}")
        logger.info(f"Tensor Parallel: {tp}, Session Length: {session_len}")
        
        # 检测是否为 Qwen2.5-VL 系列
        if 'Qwen2.5-VL' in model_path or 'Qwen2_5-VL' in model_path or model_path.startswith('Qwen/'):
            logger.info("检测到 Qwen2.5-VL 模型，使用 transformers 直接加载")
            self.use_transformers = True
            self._load_model_transformers()
        else:
            logger.info("使用 lmdeploy 加载模型")
            self._load_model_lmdeploy()
    
    def _load_model_transformers(self):
        """使用 transformers 直接加载模型（用于 Qwen2.5-VL）"""
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            
            logger.info(f"正在使用 transformers 加载模型: {self.model_path}")
            
            # 加载 processor
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            logger.info("✓ Processor 加载成功")
            
            # 加载模型
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"使用设备: {device}")
            
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True
            )
            logger.info("✓ 模型加载成功!")
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            logger.info("尝试降级到 lmdeploy 加载方式...")
            self.use_transformers = False
            self._load_model_lmdeploy()
    
    def _load_model_lmdeploy(self):
        """使用 lmdeploy 加载模型（用于其他模型）"""
        try:
            from lmdeploy import pipeline, PytorchEngineConfig
            
            backend_config = PytorchEngineConfig(
                session_len=self.session_len,
                tp=self.tp
            )
            
            logger.info(f"正在使用 lmdeploy 加载模型: {self.model_path}")
            self.pipe = pipeline(self.model_path, backend_config=backend_config)
            logger.info("✓ 模型加载成功!")
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def _load_image(self, image_path: str):
        """加载图像"""
        if self.use_transformers:
            # transformers 方式：直接使用 PIL
            return Image.open(image_path).convert('RGB')
        else:
            # lmdeploy 方式
            from lmdeploy.vl import load_image
            return load_image(image_path)
    
    def _load_images(self, image_paths: List[str]) -> List[Any]:
        """加载多张图像（用于RGB+Depth等场景）"""
        images = []
        for path in image_paths:
            if path:
                try:
                    images.append(self._load_image(path))
                except Exception as e:
                    logger.error(f"加载图像失败 {path}: {e}")
        return images
    
    def inference_batch(
        self, 
        prompts: List[Tuple[str, Any]], 
        max_retries: int = 3
    ) -> List[str]:
        """批量推理"""
        if self.use_transformers:
            return self._inference_batch_transformers(prompts, max_retries)
        else:
            return self._inference_batch_lmdeploy(prompts, max_retries)
    
    def _inference_batch_transformers(
        self,
        prompts: List[Tuple[str, Any]],
        max_retries: int = 3
    ) -> List[str]:
        """使用 transformers 进行批量推理"""
        if not self.model or not self.processor:
            raise RuntimeError("模型未加载")
        
        for attempt in range(max_retries):
            try:
                logger.debug(f"开始批量推理 (transformers)，batch size: {len(prompts)}")
                
                batch_responses = []
                for prompt_text, image_data in prompts:
                    # 处理单图或多图输入
                    # image_data 可以是单个图像或图像列表
                    if image_data is None:
                        images_list = []
                    elif isinstance(image_data, list):
                        images_list = image_data  # 多图（如 RGB + Depth）
                    else:
                        images_list = [image_data]  # 单图
                    
                    # 构建消息格式
                    content = []
                    for img in images_list:
                        if img is not None:
                            content.append({"type": "image", "image": img})
                    content.append({"type": "text", "text": prompt_text})
                    
                    messages = [
                        {
                            "role": "user",
                            "content": content,
                        }
                    ]
                    
                    # 应用聊天模板
                    text = self.processor.apply_chat_template(
                        messages, 
                        tokenize=False, 
                        add_generation_prompt=True
                    )
                    
                    # 处理图像和文本
                    if images_list:
                        inputs = self.processor(
                            text=[text],
                            images=images_list,
                            return_tensors="pt",
                            padding=True
                        )
                    else:
                        inputs = self.processor(
                            text=[text],
                            return_tensors="pt",
                            padding=True
                        )
                    
                    # 保存 input_ids 长度用于后续裁剪
                    input_ids_len = inputs.input_ids.shape[1]
                    
                    # 移到GPU
                    if torch.cuda.is_available():
                        inputs = inputs.to("cuda")
                    
                    # 生成
                    with torch.no_grad():
                        generated_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=128
                        )
                    
                    # 解码（跳过输入的 tokens）
                    generated_ids_trimmed = [
                        out_ids[input_ids_len:]
                        for out_ids in generated_ids
                    ]
                    output_text = self.processor.batch_decode(
                        generated_ids_trimmed,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False
                    )[0]
                    
                    batch_responses.append(output_text)
                
                logger.debug(f"批量推理成功，获得 {len(batch_responses)} 个响应")
                return batch_responses
                
            except Exception as e:
                logger.warning(f"推理失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    logger.error(f"推理失败，已达到最大重试次数: {e}")
                    return [f"Error: {e}"] * len(prompts)
                torch.cuda.empty_cache()
    
    def _inference_batch_lmdeploy(
        self,
        prompts: List[Tuple[str, Any]],
        max_retries: int = 3
    ) -> List[str]:
        """使用 lmdeploy 进行批量推理"""
        if not self.pipe:
            raise RuntimeError("模型未加载")
        
        for attempt in range(max_retries):
            try:
                logger.debug(f"开始批量推理 (lmdeploy)，batch size: {len(prompts)}")
                
                responses = self.pipe(prompts)
                
                if isinstance(responses, list):
                    batch_responses = [
                        r.text if hasattr(r, 'text') else str(r) 
                        for r in responses
                    ]
                else:
                    batch_responses = [
                        responses.text if hasattr(responses, 'text') else str(responses)
                    ]
                
                logger.debug(f"批量推理成功，获得 {len(batch_responses)} 个响应")
                return batch_responses
                
            except Exception as e:
                logger.warning(f"推理失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    logger.error(f"推理失败，已达到最大重试次数: {e}")
                    return [f"Error: {e}"] * len(prompts)
                torch.cuda.empty_cache()
    
    def inference_single(
        self, 
        prompt_text: str, 
        image_path: Optional[str] = None,
        depth_path: Optional[str] = None,
        max_retries: int = 3
    ) -> str:
        """
        单条推理
        
        Args:
            prompt_text: 提示文本
            image_path: 图像路径（可选）
            depth_path: 深度图路径（可选）
            max_retries: 最大重试次数
            
        Returns:
            响应文本
        """
        images = []
        if image_path:
            try:
                images.append(self._load_image(image_path))
            except Exception as e:
                logger.error(f"加载图像失败 {image_path}: {e}")
                return f"Error: Failed to load image - {e}"
        
        if depth_path:
            try:
                images.append(self._load_image(depth_path))
            except Exception as e:
                logger.error(f"加载深度图失败 {depth_path}: {e}")
                return f"Error: Failed to load depth image - {e}"
        
        # 如果有多张图，传递列表；如果只有一张，传递单个对象；如果没有，传递None
        image_data = images if len(images) > 1 else (images[0] if images else None)
        
        prompts = [(prompt_text, image_data)]
        responses = self.inference_batch(prompts, max_retries)
        return responses[0] if responses else "Error: No response"
    
    def evaluate_sample(
        self,
        question: str,
        image_path: str,
        depth_path: Optional[str] = None,
        answer: Optional[str] = None,
        system_prompt: str = "",
    ) -> Tuple[int, str]:
        """
        评估单个样本
        
        注意：只有在system_prompt明确要求评估答案时，才应该传入answer参数
        否则模型只根据问题和图片生成回答和评分
        
        Args:
            question: 问题文本
            image_path: 图像路径
            depth_path: 深度图路径（可选）
            answer: 答案文本（可选，仅在include_answer=True时使用）
            system_prompt: 系统提示词
            
        Returns:
            (score, model_response) - 分数和模型生成的回答
        """
        # 构建提示：只有明确传入answer时才包含答案
        # 这确保模型在不应该看到答案时，会根据图片和问题生成自己的回答
        if answer:
            prompt_text = f"{question}\n\nExpected answer: {answer}"
        else:
            # 不包含答案，让模型根据图片和问题生成回答
            prompt_text = question
        
        full_prompt = system_prompt + prompt_text
        
        # 推理 - 模型会返回评分和/或回答
        response = self.inference_single(full_prompt, image_path, depth_path)
        
        # 提取分数
        try:
            # 尝试提取数字分数
            score = int(response.strip().split()[0])
            if not (0 <= score <= 10):
                logger.warning(f"分数超出范围: {score}, 使用原始响应")
                score = -1
        except (ValueError, IndexError):
            logger.warning(f"无法解析分数: {response}")
            score = -1
        
        return score, response
    
    def evaluate_batch(
        self,
        samples: List[Dict[str, Any]],
        system_prompt: str = "",
        include_answer: bool = False
    ) -> List[Dict[str, Any]]:
        """
        批量评估样本
        
        Args:
            samples: 样本列表，每个样本包含 question, image_path, answer (可选)
            system_prompt: 系统提示词
            include_answer: 是否在评估中包含答案
            
        Returns:
            评估结果列表，每个结果包含 score, raw_response
        """
        # 准备批量推理数据
        prompts = []
        valid_samples = []
        
        for sample in samples:
            question = sample.get('question', '')
            image_path = sample.get('image_path', '')
            depth_path = sample.get('depth_path', '')
            answer = sample.get('answer', '') if include_answer else None
            
            if not question or not image_path:
                logger.warning(f"跳过无效样本: {sample}")
                continue
            
            # 构建提示
            if answer and include_answer:
                prompt_text = f"{question}\n\nExpected answer: {answer}"
            else:
                prompt_text = question
            
            full_prompt = system_prompt + prompt_text
            
            # 加载图像
            try:
                images = []
                images.append(self._load_image(image_path))
                
                # 如果有深度图，也加载
                if depth_path:
                    images.append(self._load_image(depth_path))
                
                # 如果有多张图，传递列表；否则传递单个对象
                image_data = images if len(images) > 1 else images[0]
                
                prompts.append((full_prompt, image_data))
                valid_samples.append(sample)
            except Exception as e:
                logger.error(f"加载图像失败 {image_path}: {e}")
                # 添加错误结果
                sample['score'] = -1
                sample['raw_response'] = f"Error: {e}"
                sample['error'] = str(e)
        
        # 批量推理
        if prompts:
            responses = self.inference_batch(prompts)
            
            # 解析结果
            for sample, response in zip(valid_samples, responses):
                try:
                    # 提取分数
                    score = int(response.strip().split()[0])
                    if not (0 <= score <= 10):
                        score = -1
                except (ValueError, IndexError):
                    score = -1
                
                sample['score'] = score
                sample['raw_response'] = response
        
        return samples
    
    def __del__(self):
        """清理资源"""
        if self.pipe:
            del self.pipe
        if self.model:
            del self.model
        if self.processor:
            del self.processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("VLM Agent 资源已释放")

