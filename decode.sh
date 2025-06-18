#! /bin/bash

# device=(0,1,2,3,4,5,6,7)
# gpu_num=8
device=(0)
gpu_num=1


export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export TORCH_COMPILE=0
export DISABLE_TORCH_COMPILE=1
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL


export CUDA_VISIBLE_DEVICES=${device[@]}
python llama3.1_peft_lora_predict.py \
    --eval_data "/linzhihang/zhangyuhao/zhanchen/ACLlama_test_data/libri_test.json" \
    --audio_tower "/linzhihang/LLMs/whisper-v3" \
    --base_model_path "/linzhihang/zhangyuhao/zhanchen/ACLlama_output/ACLlama_lora_finetune_add_clip_contrastive_loss/checkpoint-1" \
    --peft_model_id "/linzhihang/zhangyuhao/zhanchen/ACLlama_output/ACLlama_lora_finetune_add_clip_contrastive_loss/" \
    --clean_out_path "/linzhihang/zhangyuhao/zhanchen/ACLlama_output/ACLlama_lora_finetune_add_clip_contrastive_loss/test_clean.txt" \
    --other_out_path "/linzhihang/zhangyuhao/zhanchen/ACLlama_output/ACLlama_lora_finetune_add_clip_contrastive_loss/test_other.txt" \
    --num_threads ${gpu_num}


    # --eval_data "/data/s50042884/huggingface_model/libri_test_clean.json" \
    # --eval_data "/data/s50042884/huggingface_model/libri_test_other.json" \
