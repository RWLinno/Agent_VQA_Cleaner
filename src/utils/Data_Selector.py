import os
from lmdeploy import pipeline, PytorchEngineConfig
from lmdeploy.vl import load_image
import json
import torch
import argparse

system_prompt = \
"""
You are an expert annotator. Evaluate the image and the question (related to point or bounding-box annotation) together and give a single overall score from 0 to 10, where:
10 = Excellent: Image is clear and realistic, objects are well-defined, and the question is precise and unambiguous.
0 = Unusable: Image or question is too poor to allow reliable annotation.

Consider the following criteria when forming your judgment:

Image Quality:
Is the image realistic and clear?
Is the lighting appropriate?
Are the objects in the image well-defined and visible?
Are the objects in the image realistic in terms of shape and color? 
Is the object asked in the question well-defined, visible and with clear boundary? 

Question Quality:
Can the object specified in the question be clearly identified in the image?
If the question is about a part of the object, is that part clearly visible in the image?

Scoring Logic
10 is the best, 1 is the worst.
Critical Failures (force score ≤ 3):
Objects look unrealistic or have unnatural colors.
The object mentioned in the question cannot be clearly identified.
The relevant object part is completely occluded.


Please only provide the final score, only give me one number.
The question is: 
"""

import gc

tp=len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
print(f'tp={tp}')

model = '/path/to/your/model/InternVL3_5-38B'
pipe = pipeline(model, backend_config=PytorchEngineConfig(session_len=16000, tp=tp))

def batch_inference_from_json(json_file, root_path, batch_size=4, start=0, end=None, include_answer=False, output_file=None, save_per=None):
    """
    批量推理函数
    
    Args:
        json_file: JSON文件路径
        root_path: 图片根目录
        batch_size: 批量大小
        start: 开始索引
        end: 结束索引
        include_answer: 是否包含答案
        output_file: 输出文件路径
        save_per: 每处理多少条保存一次
    
    Returns:
        all_responses: 所有响应
        results: 结果列表
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if end is None:
        end = len(data)
    print(f"start={start}, end={end}, total data length={len(data)}")

    data = data[start:end]
    
    all_responses = []
    results = []
    
    for i in range(0, len(data), batch_size):
        batch_data = data[i:i + batch_size]
        
        batch_prompts = []
        batch_items = []
        for item in batch_data:
            question = None
            answer = None
            for conv in item['conversations']:
                if conv['from'] == 'human':
                    question = conv['value'].replace('<image>\n', '').replace('<image>', '').strip()
                elif conv['from'] == 'gpt':
                    answer = conv['value'].strip()
            
            if question:
                if include_answer and answer:
                    prompt_text = f"{question}\n\nExpected answer: {answer}"
                else:
                    prompt_text = question
                
                image_path = os.path.join(root_path, item['image'])
                try:
                    image = load_image(image_path)
                    batch_prompts.append((system_prompt+prompt_text, image))
                    batch_items.append({
                        'question': question,
                        'answer': answer,
                        'image_path': item['image'],
                        'full_image_path': image_path
                    })
                except Exception as e:
                    print(f"Warning: Failed to load {image_path}: {e}")
                    batch_prompts.append((system_prompt+prompt_text + " (Image not available)", None))
                    batch_items.append({
                        'question': question,
                        'answer': answer,
                        'image_path': item['image'],
                        'full_image_path': image_path,
                        'error': str(e)
                    })

        try:
            import time
            start_time=time.time()
            print("**********************")
            print(len(batch_prompts))
            print("**********************")

            response = pipe(batch_prompts)
            end_time=time.time()
            print(f"Time taken: {end_time-start_time} seconds")
            
            if isinstance(response, list):
                batch_responses = [r.text if hasattr(r, 'text') else str(r) for r in response]
            else:
                batch_responses = [response.text if hasattr(response, 'text') else str(response)]
            
            for j, (resp, item_info) in enumerate(zip(batch_responses, batch_items)):
                try:
                    score = int(resp.strip()) if resp.strip().isdigit() else resp.strip()
                except ValueError:
                    score = resp.strip()
                
                result_item = {
                    'question': item_info['question'],
                    'answer': item_info.get('answer', ''),
                    'image_path': item_info['image_path'],
                    'full_image_path': item_info['full_image_path'],
                    'score': score,
                    'raw_response': resp
                }
                
                if 'error' in item_info:
                    result_item['error'] = item_info['error']
                
                results.append(result_item)
            
            all_responses.extend(batch_responses)
            print(f"Batch {i//batch_size + 1} completed: {len(batch_responses)} responses")

            if save_per and len(results)%save_per==0:
                if output_file:
                    output_file_ = output_file.replace('.json', f"_start{start}_end{end}_gpu{os.environ['CUDA_VISIBLE_DEVICES'].replace(',', '_')}_step{len(results)}.json")
                    try:
                        with open(output_file_, 'w', encoding='utf-8') as f:
                            json.dump(results, f, ensure_ascii=False, indent=2)
                        print(f"Results saved to {output_file_}")
                    except Exception as e:
                        print(f"Error saving results to {output_file_}: {e}")
            
        except Exception as e:
            print(f"Error in batch {i//batch_size + 1}: {e}")
            error_responses = [f"Error: {e}"] * len(batch_prompts)
            all_responses.extend(error_responses)
            
            for j, item_info in enumerate(batch_items):
                result_item = {
                    'question': item_info['question'],
                    'answer': item_info.get('answer', ''),
                    'image_path': item_info['image_path'],
                    'full_image_path': item_info['full_image_path'],
                    'score': None,
                    'raw_response': error_responses[j],
                    'processing_error': str(e)
                }
                results.append(result_item)
    
    if output_file:
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"Results saved to {output_file}")
        except Exception as e:
            print(f"Error saving results to {output_file}: {e}")
    
    return all_responses, results


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--save_per", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    import datetime
    start_time = datetime.datetime.now()
    json_file = "/path/to/your/dataset/train.json"
    root_path = "/path/to/your/dataset/images"
    args = parse_args()
    batch_size = args.batch_size
    print(f"batch_size={batch_size}")

    start_idx = args.start
    end_idx = args.end
    save_per = args.save_per
    include_answer = False
    output_file = "output_files/evaluation_results.json"
    
    responses, results = batch_inference_from_json(json_file, root_path, batch_size, start_idx, end_idx, include_answer, output_file, save_per)
    
    print(f"\nGot {len(responses)} responses and {len(results)} result records:")
    for i, (resp, result) in enumerate(zip(responses[:5], results[:5])):
        print(f"Response {i+1}: {resp}")
        print(f"Score: {result['score']}")
        print(f"Question: {result['question'][:100]}...")
        print(f"Image: {result['image_path']}")
        print("-" * 50)
    end_time = datetime.datetime.now()
    print(f"batch_size={batch_size}, Total time taken: {end_time - start_time}")

    