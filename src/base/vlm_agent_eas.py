"""
VLM Agent EAS - 阿里云EAS模型部署服务代理
通过阿里云EAS API调用大模型进行推理
支持多模态输入（图像+文本）
"""
import os
import base64
import requests
import json
import logging
from typing import List, Tuple, Dict, Any, Optional
from PIL import Image
import io

logger = logging.getLogger(__name__)


class VLMAgentEAS:
    """
    阿里云EAS VLM Agent
    通过HTTP API调用EAS部署的大模型
    """
    
    def __init__(
        self,
        base_url: str,
        token: str,
        model_name: str = "Qwen3-VL-235B-A22B-Instruct-FP8",
        max_tokens: int = 256,
        timeout: int = 60
    ):
        """
        初始化EAS VLM Agent
        
        Args:
            base_url: EAS服务基础URL
            token: 认证token
            model_name: 模型名称
            max_tokens: 最大生成token数
            timeout: 请求超时时间（秒）
        """
        self.base_url = base_url.rstrip('/')
        self.api_url = f"{self.base_url}/v1/chat/completions"
        self.token = token
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.timeout = timeout
        
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }
        
        logger.info(f"初始化 EAS VLM Agent: {self.model_name}")
        logger.info(f"API URL: {self.api_url}")
        logger.info(f"Max tokens: {self.max_tokens}, Timeout: {self.timeout}s")
    
    def _image_to_base64(self, image_path: str) -> str:
        """
        将图像文件转换为base64编码
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            base64编码的图像字符串
        """
        try:
            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                buffer = io.BytesIO()
                img.save(buffer, format='JPEG', quality=95)
                img_bytes = buffer.getvalue()
                
                # 转换为base64
                img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                return f"data:image/jpeg;base64,{img_base64}"
        except Exception as e:
            logger.error(f"图像转base64失败 {image_path}: {e}")
            raise
    
    def _build_message_content(
        self,
        prompt_text: str,
        image_path: Optional[str] = None,
        depth_path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        构建消息内容，支持多图输入
        
        Args:
            prompt_text: 提示文本
            image_path: RGB图像路径
            depth_path: 深度图路径（可选）
            
        Returns:
            消息内容列表
        """
        content = []
        
        if image_path:
            try:
                img_base64 = self._image_to_base64(image_path)
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": img_base64
                    }
                })
            except Exception as e:
                logger.error(f"处理RGB图像失败: {e}")
        
        if depth_path:
            try:
                depth_base64 = self._image_to_base64(depth_path)
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": depth_base64
                    }
                })
            except Exception as e:
                logger.error(f"处理深度图失败: {e}")
        
        content.append({
            "type": "text",
            "text": prompt_text
        })
        
        return content
    
    def _call_api(
        self,
        messages: List[Dict[str, Any]],
        max_retries: int = 3
    ) -> str:
        """
        调用EAS API
        
        Args:
            messages: 消息列表
            max_retries: 最大重试次数
            
        Returns:
            模型响应文本
        """
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "stream": False
        }
        
        for attempt in range(max_retries):
            try:
                logger.debug(f"调用EAS API (尝试 {attempt + 1}/{max_retries})")
                
                response = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=self.timeout
                )
                
                logger.debug(f"API响应状态码: {response.status_code}")
                
                if response.status_code == 200:
                    result = response.json()
                    # 提取响应文本
                    if 'choices' in result and len(result['choices']) > 0:
                        content = result['choices'][0]['message']['content']
                        return content.strip()
                    else:
                        logger.error(f"API响应格式异常: {result}")
                        return f"Error: Invalid response format"
                else:
                    error_msg = f"API请求失败 (状态码: {response.status_code})"
                    logger.error(f"{error_msg}, 响应: {response.text}")
                    if attempt < max_retries - 1:
                        logger.info(f"等待重试...")
                        continue
                    return f"Error: {error_msg}"
                    
            except requests.exceptions.Timeout:
                logger.warning(f"API请求超时 (尝试 {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    continue
                return "Error: API request timeout"
                
            except Exception as e:
                logger.error(f"API调用异常 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    continue
                return f"Error: {str(e)}"
        
        return "Error: Max retries exceeded"
    
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
        try:
            content = self._build_message_content(prompt_text, image_path, depth_path)
            
            messages = [
                {"role": "system", "content": "You are a multi-modal VQA Data Quality Assessment Expert, able to accurately assess the quality of image and dialogue data."},
                {"role": "user", "content": content}
            ]
            
            # 调用API
            response = self._call_api(messages, max_retries)
            return response
            
        except Exception as e:
            logger.error(f"推理失败: {e}")
            return f"Error: {str(e)}"
    
    def inference_batch(
        self,
        prompts: List[Tuple[str, Any]],
        max_retries: int = 3
    ) -> List[str]:
        """
        批量推理（EAS API通常不支持真正的批量，这里使用顺序调用）
        
        Args:
            prompts: 提示列表，每个元素为 (prompt_text, image_data)
                    image_data 可以是 None, 图像路径字符串, 或 [图像路径, 深度图路径] 列表
            max_retries: 最大重试次数
            
        Returns:
            响应文本列表
        """
        responses = []
        
        for i, (prompt_text, image_data) in enumerate(prompts):
            logger.debug(f"处理批量推理 {i+1}/{len(prompts)}")
            
            # 解析image_data
            image_path = None
            depth_path = None
            
            if image_data is not None:
                if isinstance(image_data, str):
                    # 单个图像路径
                    image_path = image_data
                elif isinstance(image_data, list) and len(image_data) > 0:
                    # 图像列表（RGB + Depth）
                    image_path = image_data[0]
                    if len(image_data) > 1:
                        depth_path = image_data[1]
            
            # 单条推理
            response = self.inference_single(
                prompt_text=prompt_text,
                image_path=image_path,
                depth_path=depth_path,
                max_retries=max_retries
            )
            responses.append(response)
        
        return responses
    
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
            prompt_text = f"{system_prompt}{question}\n\nExpected answer: {answer}"
        else:
            # 不包含答案，让模型根据图片和问题生成回答
            prompt_text = f"{system_prompt}{question}"
        
        # 推理 - 模型会返回评分和/或回答
        response = self.inference_single(prompt_text, image_path, depth_path)
        
        # 提取分数
        try:
            # 尝试提取数字分数
            score_str = response.strip().split()[0]
            score = int(score_str)
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
                prompt_text = f"{system_prompt}{question}\n\nExpected answer: {answer}"
            else:
                prompt_text = f"{system_prompt}{question}"
            
            # 准备图像数据
            if depth_path:
                image_data = [image_path, depth_path]
            else:
                image_data = image_path
            
            prompts.append((prompt_text, image_data))
            valid_samples.append(sample)
        
        # 批量推理
        if prompts:
            responses = self.inference_batch(prompts)
            
            # 解析结果
            for sample, response in zip(valid_samples, responses):
                try:
                    # 提取分数
                    score_str = response.strip().split()[0]
                    score = int(score_str)
                    if not (0 <= score <= 10):
                        score = -1
                except (ValueError, IndexError):
                    score = -1
                
                sample['score'] = score
                sample['raw_response'] = response
        
        return samples
    
    def __del__(self):
        """清理资源"""
        logger.info("EAS VLM Agent 资源已释放")

