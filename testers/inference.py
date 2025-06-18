import os
import json
from peft import PeftModel, PeftConfig, LoraModel, LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoConfig, WhisperProcessor
import librosa
import argparse
from datasets import load_dataset
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ACLlama_el import ACLlamaForCausalLM
import torch
import random
from tqdm import tqdm
import torch.multiprocessing as mp

os.environ["TOKENIZERS_PARALLELISM"] = "true"


class BasicSetting:
    def __init__(self):
        # self.devices = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
        self.devices = ["cuda:0"]
        self.sampling_rate = 16000
        self.audio_token_len = 1  # 1500 = 300 token x 5 compress
        self.stop = "</s>"


CONFIG = BasicSetting()


def get_result(model_inputs, model, tokenizer, audio_feat):
    model.generation_config.pad_token_id = tokenizer.eos_token_id
    output_ids = model.generate(
        **model_inputs,
        audios=audio_feat,
        max_new_tokens=512,
        eos_token_id=tokenizer.eos_token_id,
        # do_sample=False,
    )
    # print(f"output_ids is : {output_ids}")
    # exit(0)
    # print(tokenizer.batch_decode(output_ids))
    input_ids = model_inputs["input_ids"]
    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]

    outputs = outputs.strip()
    if outputs.endswith(CONFIG.stop):
        outputs = outputs[:-len(CONFIG.stop)]
    outputs = outputs.strip()

    return outputs


def gen_model_inputs(tokenizer, system, prompt, device, audio_placeholder_ids, begin_of_text_id, start_header_id,
                     end_header_id, eot_id, nl_tokens, _system, _user, _assistant):
    input_ids = []
    # batch 1
    input_id = []
    system = [begin_of_text_id] + [start_header_id] + _system + [end_header_id] + nl_tokens + tokenizer(
        system).input_ids + [eot_id]
    input_id += system
    # input_id += audio_placeholder_ids
    # user_input_id = [start_header_id] + _user + [end_header_id] + nl_tokens + tokenizer(prompt).input_ids + [eot_id]
    user_input_id = [start_header_id] + _user + [end_header_id] + audio_placeholder_ids + tokenizer(
        prompt).input_ids + [eot_id]
    assistant_input_id = [start_header_id] + _assistant + [end_header_id] + nl_tokens
    input_id += user_input_id
    input_id += assistant_input_id
    input_ids.append(input_id)
    input_ids = torch.tensor(input_ids, dtype=torch.int).to(device)
    attention_mask = input_ids.ne(tokenizer.pad_token_id)
    return dict(input_ids=input_ids, attention_mask=attention_mask)


def process_items(thread_id, subset, args, CONFIG, return_dict):
    device = CONFIG.devices[thread_id % len(CONFIG.devices)]  # 根据线程ID选择设备
    print(f"Thread-{thread_id} running on {device}")

    import glob
    from safetensors.torch import load_file
    shard_files = sorted(glob.glob(os.path.join(args.base_model_path, "adapter_model-*.safetensors")))
    if not shard_files:
        shard_files = sorted(glob.glob(os.path.join(args.base_model_path, "adapter_model.safetensors")))
    need_combined_weights = {}
    for shard in shard_files:
        shard_state = load_file(shard)
        need_combined_weights.update(shard_state)
    print(f"need_combined_weights is : {need_combined_weights.keys()}")

    quantization_config = None
    model = ACLlamaForCausalLM.from_pretrained(
        args.base_model_path,
        device_map=None,
        torch_dtype=torch.float16,
        quantization_config=quantization_config,
    )
    for module in model.model.audio_tower:
        module.to(device)
    torch.cuda.empty_cache()
    # model.model.mask_tensor = model.model.mask_tensor.to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.peft_model_id)

    audio_config = model.get_model().audio_tower[0].config
    audio_config.audio_patch_token = tokenizer.get_vocab()["<audio_patch>"]
    audio_config.llm_pad_token_id = tokenizer.pad_token_id
    audio_config.audio_patch_size = CONFIG.audio_token_len

    # LoRA
    lora_config = PeftConfig.from_pretrained(args.peft_model_id)
    model = PeftModel.from_pretrained(model, args.peft_model_id, config=lora_config).to(
        dtype=torch.float16, device=device
    )
    torch.cuda.empty_cache()
    model.eval()

    DEFAULT_AUDIO_PATCH_TOKEN = "<audio_patch>"
    audio_placeholder = DEFAULT_AUDIO_PATCH_TOKEN * CONFIG.audio_token_len
    audio_placeholder = "\n" + audio_placeholder
    audio_placeholder_ids = tokenizer(audio_placeholder).input_ids

    begin_of_text_id = tokenizer.get_vocab()["<|begin_of_text|>"]
    start_header_id = tokenizer.get_vocab()["<|start_header_id|>"]
    end_header_id = tokenizer.get_vocab()["<|end_header_id|>"]
    eot_id = tokenizer.get_vocab()["<|eot_id|>"]
    nl_tokens = tokenizer('\n').input_ids
    _system = tokenizer('system').input_ids
    _user = tokenizer('user').input_ids
    _assistant = tokenizer('assistant').input_ids

    # Whisper
    audio_processor = WhisperProcessor.from_pretrained(args.audio_tower, torch_dtype=torch.float16)

    # System prompt
    system = "You are a helpful language and speech assistant. You are able to understand the speech content that the user provides, and assist the user with a variety of tasks using natural language."

    thread_results = []

    for idx, item in tqdm(subset, desc=f"Thread-{thread_id} processing"):
        question = item["instruction"]
        # 'context' 不是路径，而是包含音频数据的字典
        audio_data = item["context"]
        reference = item["answer"]

        model_inputs = gen_model_inputs(tokenizer, system, question, device, audio_placeholder_ids, begin_of_text_id,
                                        start_header_id, end_header_id, eot_id, nl_tokens, _system, _user, _assistant)

        # 直接从数据集中使用已加载的音频数组
        audio = audio_data['array']
        original_sr = audio_data['sampling_rate']

        # 如果采样率不匹配，则进行重采样
        if original_sr != CONFIG.sampling_rate:
            audio = librosa.resample(y=audio, orig_sr=original_sr, target_sr=CONFIG.sampling_rate)

        audio_feat = audio_processor(
            audio, sampling_rate=CONFIG.sampling_rate, return_tensors="pt"
        ).input_features
        audio_feat = audio_feat.to(device, dtype=torch.float16)

        predict = get_result(model_inputs, model, tokenizer, audio_feat)

        thread_results.append({
            "id": idx,
            "question": question,
            "reference": reference,
            "predict": predict
        })

    return_dict[thread_id] = thread_results


def main(args):
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 从 Hugging Face 加载数据集，缓存到 ./datasets
    print(f"Loading dataset {args.dataset_name}...")
    # 注意：数据集应包含 'instruction'、'context'（音频文件路径）和 'answer' 列。
    dataset = load_dataset(args.dataset_name, split="test", cache_dir=args.cache_dir)
    print("Dataset loaded.")

    # 如果指定，则选择数据的子集
    if args.num_samples > 0:
        print(f"Selecting first {args.num_samples} samples.")
        dataset = dataset.select(range(args.num_samples))

    items = list(dataset)

    # 用于多处理的数据分块
    if args.num_threads > 0:
        chunk_size = len(items) // args.num_threads + (1 if len(items) % args.num_threads else 0)
    else:
        chunk_size = len(items)

    subsets = [
        [(idx, items[idx]) for idx in range(i, min(i + chunk_size, len(items)))]
        for i in range(0, len(items), chunk_size)
    ]

    manager = mp.Manager()
    return_dict = manager.dict()  # 存储每个进程的结果
    processes = []

    for thread_id in range(args.num_threads):
        if thread_id >= len(subsets):
            break
        p = mp.Process(
            target=process_items,
            args=(thread_id, subsets[thread_id], args, CONFIG, return_dict),
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    # 聚合结果
    all_results = []
    for thread_id in sorted(return_dict.keys()):
        all_results.extend(return_dict[thread_id])

    # 按原始 id 对结果进行排序
    all_results.sort(key=lambda x: x['id'])

    # 确定输出文件名
    sanitized_dataset_name = args.dataset_name.replace('/', '_')
    if args.num_samples > 0:
        output_filename = f"{sanitized_dataset_name}-{args.num_samples}.json"
    else:
        output_filename = f"{sanitized_dataset_name}.json"
    
    output_path = os.path.join(args.output_dir, output_filename)

    # 保存到 JSON 文件
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)

    print(f"Processing completed! Results saved to {output_path}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--adapter_size', type=int, default=1280)
    parser.add_argument('--audio_tower', type=str, default='/huyujin/LLMs/whisper-v3')
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    parser.add_argument('--base_model_path', type=str,
                        default="/huyujin/ACLlama/ACLlama")
    parser.add_argument('--peft_model_id', type=str,
                        default="/huyujin/ACLlama/output/ACLlama_lora_mt/checkpoint-28000")
    # 用于数据集处理的新参数
    parser.add_argument('--dataset_name', type=str, required=True, 
                        help="Hugging Face Hub 上的数据集名称。它必须在 'test' 拆分中包含 'instruction'、'context' 和 'answer' 列。")
    parser.add_argument('--cache_dir', type=str, default="./datasets", 
                        help="数据集缓存目录。")
    parser.add_argument('--num_samples', type=int, default=-1, 
                        help="要从数据集中处理的样本数。如果 <= 0，则使用全部。")
    parser.add_argument('--output_dir', type=str, default="./saved", 
                        help="用于保存输出 JSON 文件的目录。")
    parser.add_argument('--num_threads', type=int, default=4)

    args = parser.parse_args()

    main(args)