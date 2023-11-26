#!/bin/bash -l
rm -rf /freespace/local/$USER/
mkdir -p /freespace/local/$USER/
cd /freespace/local/$USER/
git clone https://github.com/ggerganov/llama.cpp
git lfs install
git clone https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF
cp Llama-2-7B-Chat-GGUF/llama-2-7b-chat.Q4_K_M.gguf llama.cpp/models
cp Llama-2-7B-Chat-GGUF/config.json llama.cpp/models
