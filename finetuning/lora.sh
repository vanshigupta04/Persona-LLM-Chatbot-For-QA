datadir=/common/home/$USER/Persona-LLM-Chatbot-For-QA/finetuning/llamacpp/exp2/data/finetune_data.txt
model_path=/freespace/local/$USER/llama.cpp/models/llama-2-7b-chat.Q4_K_M.gguf
new_model_path=lama2-7b-chat-lora.gguf
/freespace/local/$USER/llama.cpp/finetune \
  --model-base $model_path \
  --train-data $datadir \
  --lora-out $new_model_path \
  --save-every 0 \
  --threads 14 \
  --ctx 4096 \
  --rope-freq-base 10000 \
  --rope-freq-scale 1.0 \
  --batch 1 \
  --grad-acc 1 \
  --adam-iter 3099 \
  --adam-alpha 0.001 \
  --lora-r 64 \
  --lora-alpha 128 \
  --use-checkpointing \
  --use-flash \
  --sample-start "\n" \
  --escape \
  --include-sample-start \
  --seed 2 \
  -ngl 35 > finetune.log 2>&1