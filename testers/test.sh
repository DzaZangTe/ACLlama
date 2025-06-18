#! /bin/bash

# device=(0,1,2,3)
# gpu_num=4
device=(0)
gpu_num=1

export HF_ENDPOINT=https://hf-mirror.com

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export TORCH_COMPILE=0
export DISABLE_TORCH_COMPILE=1
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL


export CUDA_VISIBLE_DEVICES=${device[@]}
python testers/inference.py \
    --audio_tower "/linzhihang/LLMs/whisper-v3" \
    --base_model_path "/linzhihang/zhangyuhao/zhanchen/ACLlama_output/ACLlama_lora_finetune_add_clip_contrastive_loss/checkpoint-1" \
    --peft_model_id "/linzhihang/zhangyuhao/zhanchen/ACLlama_output/ACLlama_lora_finetune_add_clip_contrastive_loss/" \
    --dataset_name "AudioLLMs/voxceleb_accent_test" \
    --num_samples 1 \
    --output_dir "./saved" \
    --num_threads ${gpu_num}