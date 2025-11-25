"""
点位验证处理器
用于：
1. 使用VLM模型生成新的点坐标（P2）
2. 与原始答案中的点（P1）进行对比
3. 使用PointQA方式验证点位准确性
4. 提取和匹配label信息
"""
import os
import re
import json
import logging
import math
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class PointVerificationProcessor:
    """点位验证处理器"""
    
    def __init__(
        self,
        processor_id: int,
        vlm_agent,
        config,
        distance_threshold: float = 50.0
    ):
        """
        初始化处理器
        
        Args:
            processor_id: 处理器ID
            vlm_agent: VLM Agent实例
            config: 配置对象
            distance_threshold: 点位距离阈值（像素）
        """
        self.processor_id = processor_id
        self.vlm_agent = vlm_agent
        self.config = config
        self.distance_threshold = distance_threshold
        
        self.processed_count = 0
        self.point_match_count = 0
        self.point_mismatch_count = 0
        
        logger.info(f"PointVerificationProcessor {processor_id} 初始化完成")
    
    def extract_label_from_question(self, question: str) -> Optional[str]:
        """
        从问题中提取label
        
        示例问题:
        - 'Point to all occurrences of "power button".'
        - 'Locate a start/pause button.'
        - 'If there are any control panel in the image?'
        
        Args:
            question: 问题文本
            
        Returns:
            提取的label，如果找不到返回None
        """
        # 模式1: 引号包围的label
        pattern1 = r'"([^"]+)"'
        match = re.search(pattern1, question)
        if match:
            return match.group(1)
        
        # 模式2: Object: xxx 格式
        pattern_object = r'Object:\s*([^\n]+)'
        match = re.search(pattern_object, question, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # 模式3: 各种动词 + label 模式
        pattern_verbs = r'(?:Locate|Find|Point(?:\s+to)?(?:\s+out)?|Show\s+me(?:\s+where)?|Generate\s+a\s+list.*?where|Where\s+are|Can\s+you\s+point\s+out)\s+(?:each|every|all|a|an|the)?\s*([a-z\s/\-]+?)(?:\s+(?:in|is|are|present)|\.|,|\?|The\s)'
        match = re.search(pattern_verbs, question, re.IGNORECASE)
        if match:
            label = match.group(1).strip()
            # 清理多余空格
            label = re.sub(r'\s+', ' ', label)
            return label
        
        # 模式4: If there (is|are) (a|any) xxx
        pattern_if = r'If\s+there\s+(?:is|are)\s+(?:a|any)\s+([a-z\s/\-]+?)(?:\s+(?:in|present)|\.|,|\?)'
        match = re.search(pattern_if, question, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # 模式5: 从常见模式提取
        common_labels = ['power button', 'start button', 'start/pause button', 
                        'control panel', 'control knob', 'door handle', 'detergent drawer']
        question_lower = question.lower()
        for label in common_labels:
            if label in question_lower:
                return label
        
        return None
    
    def extract_points_from_answer(self, answer: str) -> List[Tuple[float, float]]:
        """
        从答案中提取点坐标
        
        Args:
            answer: 答案文本（JSON格式）
            
        Returns:
            点坐标列表 [(x1, y1), (x2, y2), ...]
        """
        points = []
        
        try:
            # 尝试从JSON中提取
            # 匹配 "point_2d": [x, y] 或 [x, y] 格式
            pattern = r'\[\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\]'
            matches = re.findall(pattern, answer)
            
            for x, y in matches:
                points.append((float(x), float(y)))
            
        except Exception as e:
            logger.error(f"提取点坐标失败: {e}")
        
        return points
    
    def calculate_distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """
        计算两个点之间的欧氏距离
        
        Args:
            p1: 点1 (x1, y1)
            p2: 点2 (x2, y2)
            
        Returns:
            距离
        """
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def generate_point_with_vlm(
        self,
        question: str,
        image_path: str,
        label: str
    ) -> Tuple[Optional[Tuple[float, float]], str]:
        """
        使用VLM模型生成点坐标
        
        Args:
            question: 原始问题
            label: 提取的label
            image_path: 图像路径
            
        Returns:
            (点坐标, 模型响应) - 如果失败返回(None, error_msg)
        """
        try:
            # 构建改进的prompt，强调精确性
            prompt = f"""<image>
Task: Identify the exact location of the "{label}" in this washing machine image.

Instructions:
1. Carefully locate the CENTER POINT of the {label}
2. If it's a button, point to its center
3. If it's a control panel/display area, point to the center of the display
4. If it's a knob, point to its center
5. If it's a handle, point to the middle of the handle
6. Provide PRECISE pixel coordinates in the format [x, y]

Question: {question}

Output format: [x, y]
Example: [425, 263]

Coordinate:"""
            
            # 调用VLM推理
            response = self.vlm_agent.inference_single(
                prompt_text=prompt,
                image_path=image_path
            )
            
            # 提取点坐标
            points = self.extract_points_from_answer(response)
            
            if points:
                return points[0], response
            else:
                logger.warning(f"模型未返回有效点坐标: {response[:100]}")
                return None, response
                
        except Exception as e:
            logger.error(f"VLM生成点坐标失败: {e}")
            return None, f"Error: {str(e)}"
    
    def verify_point_with_pointqa(
        self,
        point: Tuple[float, float],
        label: str,
        image_path: str
    ) -> Tuple[bool, str, float]:
        """
        使用PointQA方式验证点位准确性，采用"反问"策略
        
        Args:
            point: 要验证的点坐标 (x, y)
            label: label名称
            image_path: 图像路径
            
        Returns:
            (is_correct, response, confidence) - 是否正确、模型响应、置信度
        """
        try:
            # 构建PointQA格式的问题，采用"反问"策略
            # 先询问点位上是什么，再对比是否是目标label
            prompt = f"""<image>
Task: Verify if the marked point [{point[0]:.2f}, {point[1]:.2f}] correctly identifies the "{label}" in this washing machine image.

Step 1: First, identify what object or component is located at point [{point[0]:.2f}, {point[1]:.2f}].
Step 2: Then, determine if this object matches the "{label}".

Please answer with:
1. What you see at the point: [describe the object at this location]
2. Verification result: [CORRECT/INCORRECT/UNCERTAIN]
3. Confidence score: [0-10]
4. Explanation: [brief reasoning]

Format:
Object at point: ...
Result: [CORRECT/INCORRECT/UNCERTAIN]
Confidence: X/10
Explanation: ..."""
            
            # 调用VLM推理
            response = self.vlm_agent.inference_single(
                prompt_text=prompt,
                image_path=image_path
            )
            
            # 解析响应
            is_correct = False
            confidence = 5.0
            
            response_upper = response.upper()
            if "CORRECT" in response_upper and "INCORRECT" not in response_upper:
                is_correct = True
            elif "INCORRECT" in response_upper:
                is_correct = False
            
            # 提取置信度
            conf_match = re.search(r'(?:Confidence|Score):\s*(\d+(?:\.\d+)?)', response, re.IGNORECASE)
            if conf_match:
                confidence = float(conf_match.group(1))
                # 归一化到0-10
                if confidence > 10:
                    confidence = confidence / 10
            
            return is_correct, response, confidence
            
        except Exception as e:
            logger.error(f"PointQA验证失败: {e}")
            return False, f"Error: {str(e)}", 0.0
    
    def process_single(
        self,
        item: Dict[str, Any],
        root_path: str
    ) -> Dict[str, Any]:
        """
        处理单个数据项
        
        Args:
            item: 数据项，包含messages和images
            root_path: 图像根路径
            
        Returns:
            处理结果
        """
        self.processed_count += 1
        
        # 解析数据
        messages = item.get('messages', [])
        images = item.get('images', [])
        item_id = item.get('id', 'unknown')
        
        if not messages or not images:
            return {
                'id': item_id,
                'error': 'Invalid data format',
                'status': 'error'
            }
        
        # 提取问题和答案
        question = ''
        answer = ''
        for msg in messages:
            if msg.get('role') == 'user':
                question = msg.get('content', '').replace('<image>', '').strip()
            elif msg.get('role') == 'assistant':
                answer = msg.get('content', '').strip()
        
        # 获取图像路径
        image_path = images[0] if isinstance(images, list) else images
        if not os.path.isabs(image_path):
            full_image_path = os.path.join(root_path, os.path.basename(image_path))
        else:
            full_image_path = image_path
        
        if not os.path.exists(full_image_path):
            # 尝试在train子目录查找
            alt_path = os.path.join(root_path, 'train', os.path.basename(image_path))
            if os.path.exists(alt_path):
                full_image_path = alt_path
            else:
                return {
                    'id': item_id,
                    'question': question,
                    'answer': answer,
                    'image_path': image_path,
                    'error': f'Image not found: {full_image_path}',
                    'status': 'error'
                }
        
        # Step 1: 提取label
        label = self.extract_label_from_question(question)
        if not label:
            logger.warning(f"无法从问题中提取label: {question[:100]}")
            label = "unknown object"
        
        # Step 2: 提取原始答案中的点（P1）
        points_original = self.extract_points_from_answer(answer)
        if not points_original:
            return {
                'id': item_id,
                'question': question,
                'answer': answer,
                'image_path': image_path,
                'full_image_path': full_image_path,
                'label': label,
                'error': 'No point coordinates found in answer',
                'status': 'no_points'
            }
        
        p1 = points_original[0]  # 取第一个点
        
        # Step 3: 使用VLM生成新的点坐标（P2）
        p2, vlm_response = self.generate_point_with_vlm(question, full_image_path, label)
        
        # Step 4: 计算P1和P2之间的距离
        distance = None
        points_match = False
        if p2:
            distance = self.calculate_distance(p1, p2)
            points_match = distance <= self.distance_threshold
            
            if points_match:
                self.point_match_count += 1
            else:
                self.point_mismatch_count += 1
        
        # Step 5: 使用PointQA验证P1的准确性
        point_is_correct, pointqa_response, pointqa_confidence = self.verify_point_with_pointqa(
            p1, label, full_image_path
        )
        
        # 构建结果
        result = {
            'id': item_id,
            'question': question,
            'answer': answer,
            'image_path': os.path.basename(image_path),
            'full_image_path': full_image_path,
            'label': label,
            
            # 原始点信息（P1）
            'point_original': {
                'coordinates': p1,
                'count': len(points_original),
                'all_points': points_original
            },
            
            # VLM推理点信息（P2）
            'point_inferred': {
                'coordinates': p2,
                'vlm_response': vlm_response,
                'success': p2 is not None
            },
            
            # 点位对比
            'point_comparison': {
                'distance': distance,
                'match': points_match,
                'threshold': self.distance_threshold
            },
            
            # PointQA验证结果
            'pointqa_verification': {
                'is_correct': point_is_correct,
                'confidence': pointqa_confidence,
                'response': pointqa_response
            },
            
            'status': 'success',
            'processor_id': self.processor_id
        }
        
        # 打印日志
        if p2:
            logger.info(
                f"[{item_id}] P1={p1}, P2={p2}, "
                f"Distance={distance:.1f}px, Match={points_match}, "
                f"PointQA={point_is_correct} (conf={pointqa_confidence:.1f})"
            )
        else:
            logger.warning(f"[{item_id}] P1={p1}, P2=None (VLM生成失败)")
        
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            统计信息字典
        """
        match_rate = (
            self.point_match_count / (self.point_match_count + self.point_mismatch_count) * 100
            if (self.point_match_count + self.point_mismatch_count) > 0
            else 0
        )
        
        return {
            'processor_id': self.processor_id,
            'processed': self.processed_count,
            'point_match': self.point_match_count,
            'point_mismatch': self.point_mismatch_count,
            'match_rate': match_rate
        }

