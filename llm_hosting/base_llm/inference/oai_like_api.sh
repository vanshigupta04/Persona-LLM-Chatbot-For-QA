#!/bin/bash -l
keep-job 30
python3 -m pip install flask
python3 /freespace/local/$USER/llama.cpp/examples/server/api_like_OAI.py