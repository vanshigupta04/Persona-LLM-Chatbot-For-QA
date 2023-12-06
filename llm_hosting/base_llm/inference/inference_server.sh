#!/bin/bash -l
#SBATCH -G 4
#SBATCH --output=inference.log
#SBATCH --time=7-00:00:00
#SBATCH --mem=512G
keep-job 30
cd /freespace/local/$USER/llama.cpp/
./server -m /freespace/local/$USER/llama.cpp/models/llama-2-7b-chat.Q4_K_M.gguf -c 1024 -ngl 35
