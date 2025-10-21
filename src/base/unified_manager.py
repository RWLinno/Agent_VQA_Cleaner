"""
UnifiedManager - 统一管理器
负责协调整个VQA数据清洗流程，管理并发处理器
"""
import os
import json
import logging
import time
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from .vlm_agent import VLMAgent
from .processor import Processor

logger = logging.getLogger(__name__)


class UnifiedManager:
    def __init__(self, config):
        """
        初始化UnifiedManager
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.vlm_agent = None
        self.processors = []
        self.results = []
        self.lock = Lock()
        
        logger.info("UnifiedManager 初始化")
        config.print_config()
    
    def initialize_agent(self):
        """初始化VLM Agent"""
        logger.info("正在初始化VLM Agent...")
        
        model_path, model_name = self.config.get_available_model()
        
        try:
            self.vlm_agent = VLMAgent(
                model_path=model_path,
                tp=self.config.TP,
                session_len=self.config.SESSION_LEN
            )
            logger.info(f"VLM Agent 初始化成功: {model_name}")
        except Exception as e:
            logger.error(f"VLM Agent 初始化失败: {e}")
            raise
    
    def create_processors(self, num_processors: Optional[int] = None):
        """
        创建多个Processor
        
        Args:
            num_processors: 处理器数量，默认使用配置中的值
        """
        if self.vlm_agent is None:
            raise RuntimeError("请先初始化VLM Agent")
        
        num_processors = num_processors or self.config.NUM_PROCESSORS
        
        logger.info(f"正在创建 {num_processors} 个 Processor...")
        
        self.processors = []
        for i in range(num_processors):
            processor = Processor(
                processor_id=i,
                vlm_agent=self.vlm_agent,
                config=self.config,
                enable_heuristic=True
            )
            self.processors.append(processor)
        
        logger.info(f"成功创建 {len(self.processors)} 个 Processor")
    
    def load_data(
        self,
        json_file: str,
        start: int = 0,
        end: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        加载JSON数据
        
        Args:
            json_file: JSON文件路径
            start: 起始索引
            end: 结束索引
            
        Returns:
            数据列表
        """
        logger.info(f"正在加载数据: {json_file}")
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if end is None:
            end = len(data)
        
        data = data[start:end]
        logger.info(f"已加载 {len(data)} 条数据 (索引 {start} 到 {end})")
        
        return data
    
    def split_data(
        self,
        data: List[Dict[str, Any]],
        num_splits: int
    ) -> List[List[Dict[str, Any]]]:
        """
        将数据分割成多个块
        
        Args:
            data: 数据列表
            num_splits: 分割数量
            
        Returns:
            分割后的数据列表
        """
        chunk_size = len(data) // num_splits
        remainder = len(data) % num_splits
        
        splits = []
        start_idx = 0
        
        for i in range(num_splits):
            end_idx = start_idx + chunk_size + (1 if i < remainder else 0)
            splits.append(data[start_idx:end_idx])
            start_idx = end_idx
        
        return splits
    
    def save_results(
        self,
        output_file: str,
        results: Optional[List[Dict[str, Any]]] = None,
        filter_out: bool = False,
        clean_format: bool = False
    ):
        """
        保存结果到文件
        
        Args:
            output_file: 输出文件路径
            results: 结果列表
            filter_out: 是否过滤掉被标记的数据
            clean_format: 是否使用清洁格式（只保留关键字段）
        """
        if results is None:
            results = self.results
        
        if filter_out:
            results = [r for r in results if not r.get('filtered', False)]
            logger.info(f"过滤后保留 {len(results)} 条数据")
        
        if clean_format:
            cleaned_results = []
            for r in results:
                cleaned = {
                    'question': r.get('question', ''),
                    'answer': r.get('answer', ''),
                    'image_path': r.get('image_path', ''),
                    'score': r.get('score', -1)
                }
                cleaned_results.append(cleaned)
            results = cleaned_results
        
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        if not output_file.endswith('.jsonl'):
            output_file = output_file.replace('.json', '.jsonl')
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
        
        logger.info(f"结果已保存到: {output_file} (共 {len(results)} 条)")
    
    def process_chunk(
        self,
        processor: Processor,
        chunk: List[Dict[str, Any]],
        root_path: str,
        include_answer: bool,
        output_file: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        处理数据块
        
        Args:
            processor: 处理器实例
            chunk: 数据块
            root_path: 图像根路径
            include_answer: 是否包含答案
            output_file: 输出文件路径
            
        Returns:
            处理结果列表
        """
        system_prompt = (
            self.config.SYSTEM_PROMPT_WITH_ANSWER if include_answer 
            else self.config.SYSTEM_PROMPT_IMAGE_QUESTION
        )
        
        batch_size = self.config.BATCH_SIZE
        chunk_results = []
        
        for i in range(0, len(chunk), batch_size):
            batch = chunk[i:i + batch_size]
            
            try:
                batch_results = processor.process_batch(
                    items=batch,
                    root_path=root_path,
                    system_prompt=system_prompt,
                    include_answer=include_answer
                )
                chunk_results.extend(batch_results)
                
                logger.info(
                    f"Processor {processor.processor_id}: "
                    f"已处理 {len(chunk_results)}/{len(chunk)}"
                )
                
                if output_file and len(chunk_results) % 50 == 0 and len(chunk_results) > 0:
                    temp_file = output_file.replace('.jsonl', f'_temp_p{processor.processor_id}_{len(chunk_results)}.jsonl')
                    try:
                        with self.lock:
                            self.save_results(temp_file, chunk_results, filter_out=False, clean_format=False)
                            logger.info(f"Processor {processor.processor_id}: 已保存中间结果到 {temp_file}")
                    except Exception as save_error:
                        logger.error(f"保存中间结果失败: {save_error}")
                
            except Exception as e:
                logger.error(f"处理批次失败: {e}")
                for item in batch:
                    try:
                        convs = item.get('conversations', [])
                        question = ''
                        answer = ''
                        if convs:
                            for conv in convs:
                                if conv.get('from') == 'human':
                                    question = conv.get('value', '').replace('<image>', '').strip()
                                    break
                            for conv in convs:
                                if conv.get('from') == 'gpt':
                                    answer = conv.get('value', '').strip()
                                    break
                        
                        image_path = item.get('image', '')
                        if isinstance(image_path, list):
                            image_path = image_path[0] if image_path else ''
                    except:
                        question = ''
                        answer = ''
                        image_path = ''
                    
                    chunk_results.append({
                        'question': question,
                        'answer': answer,
                        'image_path': image_path,
                        'score': -1,
                        'error': str(e),
                        'filtered': True,
                        'filter_reason': f"Processing error: {e}"
                    })
        
        return chunk_results
    
    def process_concurrent(
        self,
        data: List[Dict[str, Any]],
        root_path: str,
        include_answer: bool = False,
        output_file: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        并发处理数据
        
        Args:
            data: 数据列表
            root_path: 图像根路径
            include_answer: 是否包含答案
            output_file: 输出文件路径
            
        Returns:
            处理结果列表
        """
        if not self.processors:
            raise RuntimeError("没有可用的Processor，请先调用create_processors()")
        
        logger.info(f"开始并发处理 {len(data)} 条数据，使用 {len(self.processors)} 个 Processor")
        
        data_splits = self.split_data(data, len(self.processors))
        
        start_time = time.time()
        
        all_results = []
        with ThreadPoolExecutor(max_workers=len(self.processors)) as executor:
            futures = []
            for processor, chunk in zip(self.processors, data_splits):
                if len(chunk) > 0:
                    future = executor.submit(
                        self.process_chunk,
                        processor,
                        chunk,
                        root_path,
                        include_answer,
                        output_file  # 传递 output_file 以支持增量保存
                    )
                    futures.append(future)
            
            for future in as_completed(futures):
                try:
                    chunk_results = future.result()
                    with self.lock:
                        all_results.extend(chunk_results)
                        
                        if output_file and len(all_results) % self.config.SAVE_INTERVAL == 0:
                            temp_file = output_file.replace('.jsonl', f'_temp_{len(all_results)}.jsonl').replace('.json', f'_temp_{len(all_results)}.jsonl')
                            self.save_results(temp_file, all_results, filter_out=False, clean_format=False)
                            logger.info(f"已保存中间结果: {temp_file}")
                
                except Exception as e:
                    logger.error(f"收集结果时出错: {e}")
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        logger.info(f"并发处理完成，耗时: {elapsed_time:.2f} 秒")
        logger.info(f"总共处理: {len(all_results)} 条数据")
        
        self.results = all_results
        
        if output_file:
            full_file = output_file.replace('.jsonl', '_full.jsonl').replace('.json', '_full.jsonl')
            self.save_results(full_file, all_results, filter_out=False, clean_format=False)
            
            self.save_results(output_file, all_results, filter_out=True, clean_format=True)
        
        return all_results
    
    def process_sequential(
        self,
        data: List[Dict[str, Any]],
        root_path: str,
        include_answer: bool = False,
        output_file: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        顺序处理数据
        
        Args:
            data: 数据列表
            root_path: 图像根路径
            include_answer: 是否包含答案
            output_file: 输出文件路径
            
        Returns:
            处理结果列表
        """
        if not self.processors:
            raise RuntimeError("没有可用的Processor，请先调用create_processors()")
        
        logger.info(f"开始顺序处理 {len(data)} 条数据")
        
        processor = self.processors[0]
        start_time = time.time()
        
        results = self.process_chunk(processor, data, root_path, include_answer, output_file)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        logger.info(f"顺序处理完成，耗时: {elapsed_time:.2f} 秒")
        
        self.results = results
        
        if output_file:
            full_file = output_file.replace('.jsonl', '_full.jsonl').replace('.json', '_full.jsonl')
            self.save_results(full_file, results, filter_out=False, clean_format=False)
            
            self.save_results(output_file, results, filter_out=True, clean_format=True)
        
        return results
    
    def print_statistics(self):
        """打印统计信息"""
        if not self.results:
            logger.warning("没有处理结果")
            return
        
        total = len(self.results)
        filtered = sum(1 for r in self.results if r.get('filtered', False))
        kept = total - filtered
        errors = sum(1 for r in self.results if 'error' in r)
        
        scores = [r.get('score', -1) for r in self.results if r.get('score', -1) >= 0]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        print("\n" + "=" * 60)
        print("数据清洗统计信息")
        print("=" * 60)
        print(f"总处理数据: {total}")
        print(f"保留数据: {kept} ({kept/total*100:.2f}%)")
        print(f"过滤数据: {filtered} ({filtered/total*100:.2f}%)")
        print(f"错误数据: {errors}")
        print(f"平均分数: {avg_score:.2f}")
        print("=" * 60)
        
        for processor in self.processors:
            stats = processor.get_statistics()
            print(f"Processor {stats['processor_id']}: "
                  f"处理={stats['processed']}, "
                  f"保留={stats['kept']}, "
                  f"过滤={stats['filtered']}, "
                  f"错误={stats['errors']}")
        print("=" * 60 + "\n")
    
    def run(
        self,
        json_file: str,
        root_path: str,
        output_file: str,
        start: int = 0,
        end: Optional[int] = None,
        include_answer: bool = False,
        concurrent: bool = True
    ):
        """
        运行完整的数据清洗流程
        
        Args:
            json_file: 输入JSON文件路径
            root_path: 图像根路径
            output_file: 输出文件路径（.jsonl格式）
            start: 起始索引
            end: 结束索引
            include_answer: 是否在评估时包含答案
            concurrent: 是否使用并发处理
        """
        logger.info("=" * 60)
        logger.info("开始VQA数据清洗流程")
        logger.info("=" * 60)
        
        # 保存system_prompt到txt文件
        system_prompt = (
            self.config.SYSTEM_PROMPT_WITH_ANSWER if include_answer 
            else self.config.SYSTEM_PROMPT_IMAGE_QUESTION
        )
        prompt_file = output_file.replace('.jsonl', '_system_prompt.txt').replace('.json', '_system_prompt.txt')
        prompt_dir = os.path.dirname(prompt_file)
        if prompt_dir:
            os.makedirs(prompt_dir, exist_ok=True)
        with open(prompt_file, 'w', encoding='utf-8') as f:
            f.write(system_prompt)
        logger.info(f"System prompt 已保存到: {prompt_file}")
        
        self.initialize_agent()
        
        self.create_processors()
        
        data = self.load_data(json_file, start, end)
        
        if concurrent and len(self.processors) > 1:
            self.process_concurrent(data, root_path, include_answer, output_file)
        else:
            self.process_sequential(data, root_path, include_answer, output_file)
        
        self.print_statistics()
        
        logger.info("数据清洗流程完成!")

