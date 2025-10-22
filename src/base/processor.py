import os
import logging
from typing import List, Dict, Any, Optional, Callable, Tuple
from ..utils.Heuristic_Rules_Internvl2_5 import (
    check_conversations_repetition,
    flag_function_1,
    flag_function_2,
    flag_function_3,
    flag_function_4,
    flag_function_5,
    flag_function_6,
    flag_function_7
)

logger = logging.getLogger(__name__)


class Processor:
    """
    数据处理器类
    负责：
    1. 解析JSON数据
    2. 使用VLM Agent评分
    3. 应用启发式规则
    4. 过滤不合格数据
    """
    
    def __init__(
        self,
        processor_id: int,
        vlm_agent,
        config,
        enable_heuristic: bool = True
    ):
        """
        初始化Processor
        
        Args:
            processor_id: 处理器ID
            vlm_agent: VLM Agent实例
            config: 配置对象
            enable_heuristic: 是否启用启发式规则
        """
        self.processor_id = processor_id
        self.vlm_agent = vlm_agent
        self.config = config
        self.enable_heuristic = enable_heuristic
        
        self.processed_count = 0
        self.filtered_count = 0
        self.error_count = 0
        
        logger.info(f"Processor {processor_id} 初始化完成")
    
    def _find_image_path(self, root_path: str, filename: str, is_depth: bool = False) -> str:
        """
        智能查找图像文件路径，支持多种目录结构
        
        Args:
            root_path: 根路径
            filename: 文件名
            is_depth: 是否是深度图
            
        Returns:
            找到的完整路径
        """
        direct_path = os.path.join(root_path, filename)
        if os.path.exists(direct_path):
            return direct_path
        
        if is_depth:
            depth_path = os.path.join(root_path, 'depth', filename)
            if os.path.exists(depth_path):
                return depth_path
            depth_mv_path = os.path.join(root_path, 'depth_multi_view', filename)
            if os.path.exists(depth_mv_path):
                return depth_mv_path
        else:
            image_path = os.path.join(root_path, 'image', filename)
            if os.path.exists(image_path):
                return image_path
            image_mv_path = os.path.join(root_path, 'image_multi_view', filename)
            if os.path.exists(image_mv_path):
                return image_mv_path
            image_vc_path = os.path.join(root_path, 'image_visual_choice', filename)
            if os.path.exists(image_vc_path):
                return image_vc_path
        
        return direct_path
    
    def parse_conversation(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        解析conversation格式的数据
        
        Args:
            item: 原始数据项，包含 image、conversations，可选 depth
            
        Returns:
            解析后的数据，包含 question, answer, image_path, depth_path (可选)
        """
        conversations = item.get('conversations', [])
        
        image_path = item.get('image', '')
        if isinstance(image_path, list):
            image_path = image_path[0] if image_path else ''
        elif not isinstance(image_path, str):
            image_path = str(image_path)
        
        depth_path = item.get('depth', '')
        if isinstance(depth_path, list):
            depth_path = depth_path[0] if depth_path else ''
        elif not isinstance(depth_path, str):
            depth_path = str(depth_path) if depth_path else ''
        
        qa_pairs = []
        current_question = None
        
        for conv in conversations:
            if conv['from'] == 'human':
                current_question = conv['value'].replace('<image>\n', '').replace('<image>', '').strip()
            elif conv['from'] == 'gpt' and current_question:
                answer = conv['value'].strip()
                qa_pairs.append({
                    'question': current_question,
                    'answer': answer
                })
                current_question = None
        
        if qa_pairs:
            result = {
                'question': qa_pairs[0]['question'],
                'answer': qa_pairs[0]['answer'],
                'all_qa_pairs': qa_pairs,
                'image_path': image_path,
                'conversations': conversations  # 保留原始conversations用于启发式检查
            }
            if depth_path:
                result['depth_path'] = depth_path
            return result
        else:
            result = {
                'question': '',
                'answer': '',
                'all_qa_pairs': [],
                'image_path': image_path,
                'conversations': conversations
            }
            if depth_path:
                result['depth_path'] = depth_path
            return result
    
    def apply_heuristic_rules(self, item: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        应用启发式规则检查 - 增强版本，包含更多规则
        
        Args:
            item: 包含 answer 和 conversations 的数据项
            
        Returns:
            (should_keep, reasons) - 是否保留和原因列表
        """
        if not self.enable_heuristic:
            return True, []
        
        reasons = []
        answer = item.get('answer', '')
        conversations = item.get('conversations', [])
        config = self.config.HEURISTIC_CONFIG
        
        # 规则1: 检查对话中的n-gram重复
        if check_conversations_repetition(
            conversations, 
            repeat_threshold=config['repeat_threshold'],
            ngram=config['ngram']
        ):
            reasons.append("High n-gram repetition in conversations")
        
        # 规则2: 检测循环模式（超长句子+尾部重复）
        if flag_function_1(
            answer,
            super_long_words=config['super_long_sentence_words'],
            tail_len=config['tail_repeat_length'],
            tail_count=config['tail_repeat_count']
        ):
            reasons.append("Looping pattern detected (long sentence + tail repetition)")
        
        # 规则3: 检测极长句子
        if flag_function_2(answer, extreme_long_words=config['extreme_long_sentence_words']):
            reasons.append("Extremely long sentence detected")
        
        # 规则4: 检测尾部重复
        if len(answer) >= config['tail_repeat_length']:
            if flag_function_4(
                answer,
                tail_len=config['tail_repeat_length'],
                tail_count=config['tail_repeat_count']
            ):
                reasons.append("Tail repetition detected")
        
        # 规则5: 检测词汇多样性不足
        if flag_function_5(answer, min_unique_ratio=config['min_unique_word_ratio']):
            reasons.append("Insufficient vocabulary diversity")
        
        # 规则6: 检测异常特殊字符比例
        if flag_function_6(answer, max_special_ratio=config['max_special_char_ratio']):
            reasons.append("Excessive special characters")
        
        # 规则7: 检测可疑重复模式
        if flag_function_7(answer, suspicious_patterns=config['suspicious_patterns']):
            reasons.append("Suspicious repetitive pattern detected")
        
        should_keep = len(reasons) == 0
        return should_keep, reasons
    
    def process_single(
        self,
        item: Dict[str, Any],
        root_path: str,
        system_prompt: str,
        include_answer: bool = False
    ) -> Dict[str, Any]:
        """
        处理单个数据项
        
        Args:
            item: 数据项
            root_path: 图像根路径
            system_prompt: 系统提示词
            include_answer: 是否在评估时包含答案
            
        Returns:
            处理结果
        """
        self.processed_count += 1
        
        parsed = self.parse_conversation(item)
        question = parsed['question']
        answer = parsed['answer']
        image_rel_path = parsed['image_path']
        depth_rel_path = parsed.get('depth_path', '')
        
        if not isinstance(image_rel_path, str):
            logger.error(f"image_path 类型错误: {type(image_rel_path)}, 值: {image_rel_path}")
            image_rel_path = str(image_rel_path) if image_rel_path else ''
        
        full_image_path = self._find_image_path(root_path, image_rel_path, is_depth=False)
        
        full_depth_path = None
        if depth_rel_path:
            full_depth_path = self._find_image_path(root_path, depth_rel_path, is_depth=True)
        
        result = {
            'question': question,
            'answer': answer,
            'image_path': image_rel_path,
            'full_image_path': full_image_path,
            'processor_id': self.processor_id
        }
        
        if full_depth_path:
            result['depth_path'] = depth_rel_path
            result['full_depth_path'] = full_depth_path
        
        if not os.path.exists(full_image_path):
            logger.warning(f"[ERROR] {os.path.basename(image_rel_path)} | Score: -1 | Reason: Image not found")
            logger.debug(f"  Expected path: {full_image_path}")
            result['score'] = -1
            result['error'] = "Image file not found"
            result['filtered'] = True
            result['filter_reason'] = "Image not found"
            self.filtered_count += 1
            self.error_count += 1
            return result
        
        if full_depth_path and not os.path.exists(full_depth_path):
            logger.warning(f"[ERROR] {os.path.basename(depth_rel_path)} | Score: -1 | Reason: Depth image not found")
            logger.debug(f"  Expected path: {full_depth_path}")
            result['score'] = -1
            result['error'] = "Depth file not found"
            result['filtered'] = True
            result['filter_reason'] = "Depth image not found"
            self.filtered_count += 1
            self.error_count += 1
            return result
        
        if self.enable_heuristic:
            should_keep, heuristic_reasons = self.apply_heuristic_rules(parsed)
            if not should_keep:
                result['score'] = 0
                result['filtered'] = True
                result['filter_reason'] = "Heuristic rules: " + "; ".join(heuristic_reasons)
                result['heuristic_flags'] = heuristic_reasons
                self.filtered_count += 1
                image_name = os.path.basename(image_rel_path) if image_rel_path else "unknown"
                logger.warning(f"[FILTERED] {image_name} | Score: 0 | Reason: Heuristic rules ({'; '.join(heuristic_reasons[:2])}...)")
                return result
        
        try:
            score, raw_response = self.vlm_agent.evaluate_sample(
                question=question,
                image_path=full_image_path,
                depth_path=full_depth_path,
                answer=answer if include_answer else None,
                system_prompt=system_prompt
            )
            
            result['score'] = score
            result['raw_response'] = raw_response
            
            # 判断是否过滤并打印日志
            if score < self.config.SCORE_THRESHOLD:
                result['filtered'] = True
                result['filter_reason'] = f"Score below threshold ({score} < {self.config.SCORE_THRESHOLD})"
                self.filtered_count += 1
                image_name = os.path.basename(image_rel_path) if image_rel_path else "unknown"
                logger.warning(f"[FILTERED] {image_name} | Score: {score} | Threshold: {self.config.SCORE_THRESHOLD}")
            else:
                result['filtered'] = False
                image_name = os.path.basename(image_rel_path) if image_rel_path else "unknown"
                logger.info(f"[KEPT] {image_name} | Score: {score}")
            
        except Exception as e:
            logger.error(f"[ERROR] {os.path.basename(image_rel_path)} | Score: -1 | Reason: {str(e)[:100]}")
            result['score'] = -1
            result['error'] = str(e)
            result['filtered'] = True
            result['filter_reason'] = f"Evaluation error: {e}"
            self.filtered_count += 1
            self.error_count += 1
        
        return result
    
    def process_batch(
        self,
        items: List[Dict[str, Any]],
        root_path: str,
        system_prompt: str,
        include_answer: bool = False
    ) -> List[Dict[str, Any]]:
        """
        批量处理数据项
        
        Args:
            items: 数据项列表
            root_path: 图像根路径
            system_prompt: 系统提示词
            include_answer: 是否在评估时包含答案
            
        Returns:
            处理结果列表
        """
        results = []
        
        items_to_evaluate = []
        for item in items:
            parsed = self.parse_conversation(item)
            
            image_rel_path = parsed['image_path']
            if not isinstance(image_rel_path, str):
                logger.error(f"image_path 类型错误: {type(image_rel_path)}, 值: {image_rel_path}")
                image_rel_path = str(image_rel_path) if image_rel_path else ''
            
            full_image_path = self._find_image_path(root_path, image_rel_path, is_depth=False)
            
            depth_rel_path = parsed.get('depth_path', '')
            full_depth_path = None
            if depth_rel_path:
                full_depth_path = self._find_image_path(root_path, depth_rel_path, is_depth=True)
            
            if not os.path.exists(full_image_path):
                result = {
                    'question': parsed['question'],
                    'answer': parsed['answer'],
                    'image_path': image_rel_path,  # 使用已经验证过的 image_rel_path
                    'full_image_path': full_image_path,
                    'score': -1,
                    'error': "Image file not found",
                    'filtered': True,
                    'filter_reason': "Image not found",
                    'processor_id': self.processor_id
                }
                if depth_rel_path:
                    result['depth_path'] = depth_rel_path
                results.append(result)
                self.filtered_count += 1
                self.error_count += 1
                continue
            
            if full_depth_path and not os.path.exists(full_depth_path):
                result = {
                    'question': parsed['question'],
                    'answer': parsed['answer'],
                    'image_path': image_rel_path,
                    'depth_path': depth_rel_path,
                    'full_image_path': full_image_path,
                    'full_depth_path': full_depth_path,
                    'score': -1,
                    'error': "Depth file not found",
                    'filtered': True,
                    'filter_reason': "Depth image not found",
                    'processor_id': self.processor_id
                }
                results.append(result)
                self.filtered_count += 1
                self.error_count += 1
                continue
            
            if self.enable_heuristic:
                should_keep, heuristic_reasons = self.apply_heuristic_rules(parsed)
                if not should_keep:
                    result = {
                        'question': parsed['question'],
                        'answer': parsed['answer'],
                        'image_path': image_rel_path,
                        'full_image_path': full_image_path,
                        'score': 0,
                        'filtered': True,
                        'filter_reason': "Heuristic rules: " + "; ".join(heuristic_reasons),
                        'heuristic_flags': heuristic_reasons,
                        'processor_id': self.processor_id
                    }
                    results.append(result)
                    self.filtered_count += 1
                    image_name = os.path.basename(image_rel_path) if image_rel_path else "unknown"
                    logger.warning(f"[FILTERED] {image_name} | Score: 0 | Reason: Heuristic rules ({'; '.join(heuristic_reasons[:2])}...)")
                    continue
            
            eval_item = {
                'question': parsed['question'],
                'answer': parsed['answer'],
                'image_path': full_image_path,
                'image_rel_path': image_rel_path
            }
            if full_depth_path:
                eval_item['depth_path'] = full_depth_path
                eval_item['depth_rel_path'] = depth_rel_path
            items_to_evaluate.append(eval_item)
        
        if items_to_evaluate:
            try:
                evaluated = self.vlm_agent.evaluate_batch(
                    items_to_evaluate,
                    system_prompt=system_prompt,
                    include_answer=include_answer
                )
                
                for eval_result in evaluated:
                    score = eval_result.get('score', -1)
                    
                    image_rel_path = eval_result.get('image_rel_path', eval_result.get('image_path', ''))
                    
                    result = {
                        'question': eval_result['question'],
                        'answer': eval_result.get('answer', ''),
                        'image_path': image_rel_path,
                        'full_image_path': eval_result['image_path'],
                        'score': score,
                        'raw_response': eval_result.get('raw_response', ''),
                        'processor_id': self.processor_id
                    }
                    
                    if 'error' in eval_result:
                        result['error'] = eval_result['error']
                    
                    image_name = os.path.basename(image_rel_path) if image_rel_path else "unknown"
                    
                    if score < self.config.SCORE_THRESHOLD or score == -1:
                        result['filtered'] = True
                        if score == -1:
                            error_msg = eval_result.get('error', 'Unknown error')
                            result['filter_reason'] = f"Evaluation error: {error_msg}"
                            logger.error(f"[ERROR] {image_name} | Score: -1 | Reason: {str(error_msg)[:100]}")
                        else:
                            result['filter_reason'] = f"Score below threshold ({score} < {self.config.SCORE_THRESHOLD})"
                            logger.warning(f"[FILTERED] {image_name} | Score: {score} | Threshold: {self.config.SCORE_THRESHOLD}")
                        self.filtered_count += 1
                    else:
                        result['filtered'] = False
                        logger.info(f"[KEPT] {image_name} | Score: {score}")
                    
                    results.append(result)
                
            except Exception as e:
                logger.error(f"批量评估失败: {e}")
                for item in items_to_evaluate:
                    result = {
                        'question': item['question'],
                        'answer': item.get('answer', ''),
                        'image_path': item.get('image_rel_path', ''),
                        'full_image_path': item['image_path'],
                        'score': -1,
                        'error': str(e),
                        'filtered': True,
                        'filter_reason': f"Evaluation error: {e}",
                        'processor_id': self.processor_id
                    }
                    results.append(result)
                    self.filtered_count += 1
                    self.error_count += 1
        
        self.processed_count += len(items)
        return results
    
    def get_statistics(self) -> Dict[str, int]:
        """
        获取处理统计信息
        
        Returns:
            统计信息字典
        """
        return {
            'processor_id': self.processor_id,
            'processed': self.processed_count,
            'filtered': self.filtered_count,
            'kept': self.processed_count - self.filtered_count,
            'errors': self.error_count
        }

