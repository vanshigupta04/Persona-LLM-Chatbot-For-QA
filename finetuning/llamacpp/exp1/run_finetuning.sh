#!/bin/bash -l
#SBATCH -G 4
#SBATCH --output=finetune.log
#SBATCH --time=7-00:00:00
#SBATCH --mem=512G
keep-job 30
datadir=/common/home/$USER/Persona-LLM-Chatbot-For-QA/notebooks/data/friends_dataset.txt
model_path=/freespace/local/$USER/llama.cpp/models/llama-2-7b-chat.Q4_K_M.gguf
new_model_path=llama2-7b-chat-lora.gguf
cd /freespace/local/$USER/llama.cpp/
./finetune \
  --model-base $model_path \
  --train-data $datadir \
  --lora-out $new_model_path \
  --save-every 0 \
  --threads 14 \
  --ctx 256 \
  --rope-freq-base 10000 \
  --rope-freq-scale 1.0 \
  --batch 1 \
  --grad-acc 1 \
  --adam-iter 256 \
  --adam-alpha 0.001 \
  --lora-r 4 \
  --lora-alpha 4 \
  --use-checkpointing \
  --use-flash \
  --sample-start "\n" \
  --escape \
  --include-sample-start \
  --seed 1
