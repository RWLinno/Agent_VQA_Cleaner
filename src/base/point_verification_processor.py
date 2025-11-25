"""
点位验证处理器（增强版）
用于：
1. 使用VLM模型生成新的点坐标（P2）
2. 与原始答案中的点（P1）进行对比
3. 使用PointQA方式验证点位准确性
4. 提取和匹配label信息
5. 🆕 生成扰动点作为负样本进行交叉验证
6. 🆕 严格筛选：点位匹配 + PointQA正确 + 负样本识别
"""
import os
import re
import json
import logging
import math
import random
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class PointVerificationProcessor:
    """点位验证处理器"""
    
    def __init__(
        self,
        processor_id: int,
        vlm_agent,
        config,
        distance_threshold: float = 50.0,
        enable_perturbation: bool = True,
        perturbation_range: int = 100
    ):
        """
        初始化处理器
        
        Args:
            processor_id: 处理器ID
            vlm_agent: VLM Agent实例
            config: 配置对象
            distance_threshold: 点位距离阈值（像素）
            enable_perturbation: 是否启用点位扰动验证
            perturbation_range: 扰动范围（像素）
        """
        self.processor_id = processor_id
        self.vlm_agent = vlm_agent
        self.config = config
        self.distance_threshold = distance_threshold
        self.enable_perturbation = enable_perturbation
        self.perturbation_range = perturbation_range
        
        # 统计信息
        self.processed_count = 0
        self.point_match_count = 0
        self.point_mismatch_count = 0
        self.pointqa_correct_count = 0
        self.pointqa_incorrect_count = 0
        self.negative_detected_count = 0  # 正确识别负样本的数量
        self.negative_failed_count = 0    # 未能识别负样本的数量
        self.strict_pass_count = 0        # 通过严格筛选的数量
        self.strict_fail_count = 0        # 未通过严格筛选的数量
        
        logger.info(
            f"PointVerificationProcessor {processor_id} 初始化完成 "
            f"(扰动验证: {enable_perturbation}, 扰动范围: {perturbation_range}px)"
        )
    
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
    
    def generate_perturbed_point(
        self,
        original_point: Tuple[float, float],
        perturbation_range: int = None,
        image_size: Tuple[int, int] = (1000, 1000)
    ) -> Tuple[float, float]:
        """
        生成扰动点（负样本）
        
        策略：在原点周围随机扰动，确保扰动距离足够大（超过阈值）
        
        Args:
            original_point: 原始点 (x, y)
            perturbation_range: 扰动范围（像素），默认使用self.perturbation_range
            image_size: 图像尺寸 (width, height)
            
        Returns:
            扰动后的点 (x', y')
        """
        if perturbation_range is None:
            perturbation_range = self.perturbation_range
        
        x, y = original_point
        width, height = image_size
        
        # 确保扰动距离大于阈值的1.5倍，制造明显的负样本
        min_distance = self.distance_threshold * 1.5
        
        max_attempts = 50
        for _ in range(max_attempts):
            # 在perturbation_range范围内随机扰动
            dx = random.uniform(-perturbation_range, perturbation_range)
            dy = random.uniform(-perturbation_range, perturbation_range)
            
            new_x = x + dx
            new_y = y + dy
            
            # 确保在图像范围内
            new_x = max(0, min(width - 1, new_x))
            new_y = max(0, min(height - 1, new_y))
            
            # 检查距离是否足够大
            distance = self.calculate_distance(original_point, (new_x, new_y))
            if distance >= min_distance:
                return (new_x, new_y)
        
        # 如果随机方法失败，使用确定性方法：向随机方向偏移min_distance
        angle = random.uniform(0, 2 * math.pi)
        dx = min_distance * math.cos(angle)
        dy = min_distance * math.sin(angle)
        
        new_x = max(0, min(width - 1, x + dx))
        new_y = max(0, min(height - 1, y + dy))
        
        return (new_x, new_y)
    
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
        image_path: str,
        is_negative_sample: bool = False
    ) -> Tuple[bool, str, float]:
        """
        使用PointQA方式验证点位准确性（改进版）
        
        Args:
            point: 要验证的点坐标 (x, y)
            label: label名称
            image_path: 图像路径
            is_negative_sample: 是否为负样本（扰动点）
            
        Returns:
            (is_correct, response, confidence) - 是否正确、模型响应、置信度
        """
        try:
            # 改进的prompt：增加容错性和清晰度
            prompt = f"""<image>
Task: Verify if the point [{point[0]:.2f}, {point[1]:.2f}] correctly locates the "{label}" in this washing machine image.

Instructions:
- Consider the point CORRECT if it points to the {label} or very close to it (center or edge both acceptable)
- Consider the point INCORRECT if it clearly points to a different object or is far from the {label}
- Use UNCERTAIN only if the image is unclear or ambiguous

Verification Steps:
1. Look at the region around point [{point[0]:.2f}, {point[1]:.2f}]
2. Identify what component/object is there
3. Determine if it matches "{label}"

Output Format:
Result: [CORRECT/INCORRECT/UNCERTAIN]
Confidence: [0-10]
Reason: [What you see at this location and why you judge it this way]

Your Response:"""
            
            # 调用VLM推理
            response = self.vlm_agent.inference_single(
                prompt_text=prompt,
                image_path=image_path
            )
            
            # 解析响应
            is_correct = False
            confidence = 5.0
            
            response_upper = response.upper()
            
            # 更精确的解析：避免"INCORRECT"包含"CORRECT"的误判
            if "INCORRECT" in response_upper or "IN CORRECT" in response_upper:
                is_correct = False
            elif "CORRECT" in response_upper:
                is_correct = True
            elif "UNCERTAIN" in response_upper:
                # UNCERTAIN视为不正确（保守策略）
                is_correct = False
                confidence = 3.0  # 不确定时置信度较低
            
            # 提取置信度
            conf_match = re.search(r'(?:Confidence|Score):\s*(\d+(?:\.\d+)?)', response, re.IGNORECASE)
            if conf_match:
                confidence = float(conf_match.group(1))
                # 归一化到0-10
                if confidence > 10:
                    confidence = confidence / 10
            
            # 记录日志
            sample_type = "负样本" if is_negative_sample else "正样本"
            logger.debug(
                f"PointQA验证 ({sample_type}): "
                f"点=[{point[0]:.1f}, {point[1]:.1f}], "
                f"Label={label}, "
                f"结果={'正确' if is_correct else '不正确'}, "
                f"置信度={confidence:.1f}"
            )
            
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
        
        # Step 5: 使用PointQA验证P1的准确性（正样本）
        point_is_correct, pointqa_response, pointqa_confidence = self.verify_point_with_pointqa(
            p1, label, full_image_path, is_negative_sample=False
        )
        
        if point_is_correct:
            self.pointqa_correct_count += 1
        else:
            self.pointqa_incorrect_count += 1
        
        # Step 6: 生成扰动点并验证（负样本交叉验证）
        perturbed_point = None
        perturbed_is_correct = None
        perturbed_response = None
        perturbed_confidence = None
        perturbed_distance = None
        negative_sample_detected = False
        
        if self.enable_perturbation:
            try:
                # 生成扰动点P1'
                perturbed_point = self.generate_perturbed_point(p1)
                perturbed_distance = self.calculate_distance(p1, perturbed_point)
                
                # 使用PointQA验证扰动点（应该判定为不正确）
                perturbed_is_correct, perturbed_response, perturbed_confidence = self.verify_point_with_pointqa(
                    perturbed_point, label, full_image_path, is_negative_sample=True
                )
                
                # 判断是否正确识别了负样本
                negative_sample_detected = not perturbed_is_correct
                
                if negative_sample_detected:
                    self.negative_detected_count += 1
                else:
                    self.negative_failed_count += 1
                    logger.warning(
                        f"[{item_id}] 未能识别负样本: "
                        f"P1={p1}, P1'={perturbed_point}, "
                        f"距离={perturbed_distance:.1f}px, "
                        f"但PointQA判定为正确"
                    )
                
            except Exception as e:
                logger.error(f"生成或验证扰动点失败: {e}")
                negative_sample_detected = False
        
        # Step 7: 严格筛选判定
        # 同时满足以下条件才算通过：
        # 1. P1和P2距离匹配（如果P2生成成功）
        # 2. PointQA验证P1为正确
        # 3. 能正确识别负样本P1'为不正确（如果启用扰动）
        strict_pass = True
        fail_reasons = []
        
        # 条件1：点位匹配
        if p2 and not points_match:
            strict_pass = False
            fail_reasons.append(f"点位不匹配(距离={distance:.1f}px > {self.distance_threshold}px)")
        elif not p2:
            strict_pass = False
            fail_reasons.append("VLM未能生成P2点位")
        
        # 条件2：PointQA正确
        if not point_is_correct:
            strict_pass = False
            fail_reasons.append(f"PointQA判定不正确(置信度={pointqa_confidence:.1f})")
        
        # 条件3：负样本识别
        if self.enable_perturbation and not negative_sample_detected:
            strict_pass = False
            fail_reasons.append(f"未能识别负样本(P1'也被判定为正确)")
        
        if strict_pass:
            self.strict_pass_count += 1
        else:
            self.strict_fail_count += 1
        
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
            
            # PointQA验证结果（正样本）
            'pointqa_verification': {
                'is_correct': point_is_correct,
                'confidence': pointqa_confidence,
                'response': pointqa_response
            },
            
            # 扰动点验证结果（负样本）
            'perturbation_verification': {
                'enabled': self.enable_perturbation,
                'perturbed_point': perturbed_point,
                'distance_from_original': perturbed_distance,
                'is_correct': perturbed_is_correct,
                'confidence': perturbed_confidence,
                'response': perturbed_response,
                'negative_detected': negative_sample_detected
            },
            
            # 严格筛选结果
            'strict_filtering': {
                'pass': strict_pass,
                'fail_reasons': fail_reasons if not strict_pass else []
            },
            
            'status': 'success',
            'processor_id': self.processor_id
        }
        
        # 打印详细日志
        log_parts = [f"[{item_id}]"]
        log_parts.append(f"P1={p1}")
        
        if p2:
            log_parts.append(f"P2={p2}")
            log_parts.append(f"Dist={distance:.1f}px")
            log_parts.append(f"Match={points_match}")
        else:
            log_parts.append("P2=None")
        
        log_parts.append(f"PointQA={'✓' if point_is_correct else '✗'}({pointqa_confidence:.1f})")
        
        if self.enable_perturbation and perturbed_point:
            log_parts.append(f"P1'={[f'{x:.1f}' for x in perturbed_point]}")
            log_parts.append(f"NegDetect={'✓' if negative_sample_detected else '✗'}")
        
        log_parts.append(f"Strict={'✓ PASS' if strict_pass else '✗ FAIL'}")
        
        logger.info(" | ".join(log_parts))
        
        if not strict_pass:
            logger.info(f"  └─ 失败原因: {'; '.join(fail_reasons)}")
        
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取详细统计信息
        
        Returns:
            统计信息字典
        """
        # 点位匹配率
        match_rate = (
            self.point_match_count / (self.point_match_count + self.point_mismatch_count) * 100
            if (self.point_match_count + self.point_mismatch_count) > 0
            else 0
        )
        
        # PointQA正确率
        pointqa_correct_rate = (
            self.pointqa_correct_count / (self.pointqa_correct_count + self.pointqa_incorrect_count) * 100
            if (self.pointqa_correct_count + self.pointqa_incorrect_count) > 0
            else 0
        )
        
        # 负样本检测率
        negative_detect_rate = (
            self.negative_detected_count / (self.negative_detected_count + self.negative_failed_count) * 100
            if (self.negative_detected_count + self.negative_failed_count) > 0
            else 0
        )
        
        # 严格筛选通过率
        strict_pass_rate = (
            self.strict_pass_count / (self.strict_pass_count + self.strict_fail_count) * 100
            if (self.strict_pass_count + self.strict_fail_count) > 0
            else 0
        )
        
        return {
            'processor_id': self.processor_id,
            'processed': self.processed_count,
            
            # 点位对比统计
            'point_comparison': {
                'match': self.point_match_count,
                'mismatch': self.point_mismatch_count,
                'match_rate': match_rate
            },
            
            # PointQA验证统计
            'pointqa_verification': {
                'correct': self.pointqa_correct_count,
                'incorrect': self.pointqa_incorrect_count,
                'correct_rate': pointqa_correct_rate
            },
            
            # 负样本检测统计
            'negative_sample': {
                'detected': self.negative_detected_count,
                'failed': self.negative_failed_count,
                'detect_rate': negative_detect_rate
            },
            
            # 严格筛选统计
            'strict_filtering': {
                'pass': self.strict_pass_count,
                'fail': self.strict_fail_count,
                'pass_rate': strict_pass_rate
            }
        }

