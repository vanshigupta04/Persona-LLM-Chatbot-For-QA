#!/bin/bash -l
#SBATCH -G 4
#SBATCH --output=lora.log
#SBATCH --time=7-00:00:00
#SBATCH --mem=512G

datadir=/common/home/$USER/Persona-LLM-Chatbot-For-QA/data/finetune_data.txt
model_path=/freespace/local/$USER/llama.cpp/models/llama-2-7b-chat.Q4_K_M.gguf
new_model_path=lama2-7b-chat-lora.gguf
cd /freespace/local/$USER/llama.cpp/
./finetune \
  --model-base $model_path \
  --train-data $datadir \
  --lora-out $new_model_path \
  --save-every 0 \
  --threads 16 \
  --ctx 4096 \
  --rope-freq-base 10000 \
  --rope-freq-scale 1.0 \
  --batch 1 \
  --grad-acc 1 \
  --adam-iter 6198 \
  --adam-alpha 0.001 \
  --lora-r 4 \
  --lora-alpha 4 \
  --use-checkpointing \
  --use-flash \
  --sample-start "\n" \
  --escape \
  --seed 2 \
  -ngl 35 