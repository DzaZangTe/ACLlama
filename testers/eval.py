#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import json
import requests
import urllib3
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import random

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# API 配置
FOWARD_URL = "URL"
API_KEY = "KEY"
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}
MODEL = "chatgpt-4o-latest"


def request_api(message, max_retries=3):
    """发送 API 请求"""
    payload = {
        "model": MODEL,
        "messages": message,
        "temperature": 0.3,
        "max_tokens": 512,
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(FOWARD_URL, headers=HEADERS, json=payload, verify=False, timeout=120)
            response_data = response.json()
            
            if "choices" in response_data and len(response_data["choices"]) > 0:
                return response_data["choices"][0]["message"]["content"]
            else:
                print(f"Unexpected response format: {response_data}")
                
        except requests.exceptions.RequestException as e:
            print(f"API request error (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(random.uniform(1, 3))  # 随机延迟重试
    
    return None


def gpt4o_as_judge_one_sample(question, reference, prediction, eval_type="5-point"):
    """使用 GPT-4o 评估单个样本"""
    
    if eval_type == "binary":
        PROMPT_TEMPLATE = """\
[Reference Answer]
{reference}

[Model Answer]
{prediction}

[Question]
{question}

[Task]
Rate the model's answer based on its alignment with the reference answer, focusing on accuracy and relevance to the reference provided. Please be critical on the details.
Criteria: Assess if the model's response mirrors the reference in terms of content, accuracy, and relevance. Please give a score of 0 or 1. 
Score0: The answer is refusing to give concrete results, providing something like 'cannot decide'.
Score0: The answer is wrong, providing incorrect or irrelevant information compared to the reference. 
Score1: The answer is correct, capturing or covering the meaning from the reference.

Your response should be formatted as follows:
Explanation: (Provide a concise explanation of your rating, comparing the reference answer with the model's response. "The reference answer is [XXX], while the model's answer is [YYY]. I think ...")
Rating: (int)"""
    else:  # 5-point scale
        PROMPT_TEMPLATE = """\
[Reference Answer]
{reference}

[Model Answer]
{prediction}

[Question]
{question}

[Task]
Rate the model's answer based on its alignment with the reference answer, focusing on accuracy and relevance to the reference provided. Please be critical on the details. If the model response is something like 'cannot decide', please rate as 0.
Criteria: Assess if the model's response mirrors the reference in terms of content, accuracy, and relevance.
Score0: The answer is refusing to give concrete results, providing something like 'cannot decide'.
Score0: The answer is completely misaligned, providing incorrect or irrelevant information compared to the reference.
Score1: The answer shows minimal alignment, often misunderstanding or providing irrelevant details unrelated to the reference.
Score2: The answer recognizes the topic but diverges significantly from the reference in accuracy or relevance.
Score3: The answer aligns with the reference generally but lacks detail or precise accuracy in some aspects.
Score4: The answer is mostly accurate and relevant, closely following the reference but could be clearer or more detailed.
Score5: The answer is highly accurate, detailed, and matches the reference answer perfectly, capturing its essence and detail.

Your response should be formatted as follows:
Explanation: (Provide a concise explanation of your rating, comparing the reference answer with the model's response. "The reference answer is [XXX], while the model's answer is [YYY]. I think ...")
Rating: (int)"""

    evaluation_prompt = PROMPT_TEMPLATE.format(question=question, prediction=prediction, reference=reference)

    messages = [
        {"role": "user", "content": evaluation_prompt},
    ]

    output = request_api(messages)
    
    if output is None:
        output = "API request failed"
        rate_score = 0.0
        success = 0
    else:
        try:
            # 尝试从输出中提取评分
            if "Rating:" in output:
                rating_part = output.split("Rating:")[-1].strip()
                rate_score = float(rating_part.split()[0])
            else:
                # 如果没有找到 Rating:，尝试提取最后一个数字
                import re
                numbers = re.findall(r'\d+', output)
                if numbers:
                    rate_score = float(numbers[-1])
                else:
                    rate_score = 0.0
            success = 1
        except:
            rate_score = 0.0
            success = 0

    sample_rating_detail = {
        'question': question,
        'reference': reference,
        'model_prediction': prediction,
        'judge_response': output,
        'rate_score': rate_score,
        'success': success,
    }

    return sample_rating_detail


def process_single_item(item, eval_type):
    """处理单个评估项目"""
    question = item['question']
    reference = item['reference']
    prediction = item['predict']
    original_id = item['id']
    
    result = gpt4o_as_judge_one_sample(question, reference, prediction, eval_type)
    result['original_id'] = original_id
    
    return result


def evaluate_json_file(json_file_path, eval_type="5-point", max_workers=8):
    """评估保存的 JSON 文件"""
    
    # 读取 JSON 文件
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} samples from {json_file_path}")
    
    # 检查是否有进度文件
    progress_file = json_file_path.replace('.json', '_progress.jsonl')
    processed_ids = set()
    all_results = []
    
    if os.path.exists(progress_file):
        with open(progress_file, 'r', encoding='utf-8') as f:
            for line in f:
                result = json.loads(line.strip())
                all_results.append(result)
                processed_ids.add(result['original_id'])
        print(f"Resuming from progress file, {len(processed_ids)} items already processed")
    
    # 过滤未处理的数据
    remaining_data = [item for item in data if item['id'] not in processed_ids]
    
    if not remaining_data:
        print("All items have been processed!")
    else:
        print(f"Processing {len(remaining_data)} remaining items...")
        
        # 使用线程池并行处理
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            with open(progress_file, 'a', encoding='utf-8') as progress_f:
                # 提交所有任务
                future_to_item = {
                    executor.submit(process_single_item, item, eval_type): item 
                    for item in remaining_data
                }
                
                # 处理完成的任务
                for future in tqdm(as_completed(future_to_item), total=len(remaining_data), desc="Evaluating"):
                    try:
                        result = future.result()
                        all_results.append(result)
                        
                        # 实时保存进度
                        progress_f.write(json.dumps(result, ensure_ascii=False) + '\n')
                        progress_f.flush()
                        
                    except Exception as e:
                        item = future_to_item[future]
                        print(f"Error processing item {item['id']}: {e}")
    
    # 按原始 ID 排序
    all_results.sort(key=lambda x: x['original_id'])
    
    # 计算分数
    if len(all_results) == 0:
        print("Warning: No results obtained!")
        return {
            'judge_score': 0.0,
            'success_rate': 0.0,
            'total_samples': 0,
            'eval_type': eval_type
        }, []
    
    all_scores = [detail['rate_score'] for detail in all_results]
    success_rate = sum([detail['success'] for detail in all_results]) / len(all_results)
    
    if eval_type == "binary":
        avg_score = sum(all_scores) / len(all_scores) * 100  # 转换为百分比
    else:
        avg_score = sum(all_scores) / len(all_scores) * 20   # 5分制转换为百分制
    
    judge_results = {
        'judge_score': avg_score,
        'success_rate': success_rate,
        'total_samples': len(all_results),
        'eval_type': eval_type
    }
    
    return judge_results, all_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate model predictions using GPT-4o API")
    parser.add_argument('--json_file', type=str, required=True,
                        help="Path to the JSON file containing predictions")
    parser.add_argument('--eval_type', type=str, choices=['binary', '5-point'], default='5-point',
                        help="Evaluation type: binary (0-1) or 5-point (0-5)")
    parser.add_argument('--max_workers', type=int, default=8,
                        help="Number of concurrent API requests")
    parser.add_argument('--output_dir', type=str, default='./eval_results',
                        help="Directory to save evaluation results")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 执行评估
    print(f"Starting evaluation with {args.eval_type} scoring using GPT-4o API...")
    judge_results, all_details = evaluate_json_file(
        args.json_file, 
        args.eval_type, 
        args.max_workers
    )
    
    # 保存结果
    json_filename = os.path.basename(args.json_file).replace('.json', '')
    
    # 保存评估摘要
    summary_file = os.path.join(args.output_dir, f"{json_filename}_gpt4o_eval_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(judge_results, f, indent=4, ensure_ascii=False)
    
    # 保存详细结果
    details_file = os.path.join(args.output_dir, f"{json_filename}_gpt4o_eval_details.json")
    with open(details_file, 'w', encoding='utf-8') as f:
        json.dump(all_details, f, indent=4, ensure_ascii=False)
    
    # 打印结果
    print(f"\n=== GPT-4o Evaluation Results ===")
    print(f"Total samples: {judge_results['total_samples']}")
    print(f"Average score: {judge_results['judge_score']:.2f}")
    print(f"Success rate: {judge_results['success_rate']:.2f}")
    print(f"Evaluation type: {judge_results['eval_type']}")
    print(f"Results saved to: {args.output_dir}")
    
    # 清理进度文件
    progress_file = args.json_file.replace('.json', '_progress.jsonl')
    if os.path.exists(progress_file):
        os.remove(progress_file)
        print("Progress file cleaned up")


if __name__ == "__main__":
    main()