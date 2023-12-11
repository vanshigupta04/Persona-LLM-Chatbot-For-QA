#!/bin/bash -l
#SBATCH -G 4
#SBATCH --output=inference.log
#SBATCH --time=7-00:00:00
#SBATCH --mem=512G
keep-job 30
/freespace/local/$USER/llama.cpp/server -m /freespace/local/$USER/llama.cpp/models/llama-2-7b-chat.Q4_K_M.gguf -c 1024 -ngl 0 -t 14 --port 8081 --lora /common/home/$USER/Persona-LLM-Chatbot-For-QA/finetuning/adapters/llama-2-7b-chat-qlora/ggml-adapter-model.bin --lora-base /freespace/local/$USER/Llama-2-7b-chat-f16.gguf