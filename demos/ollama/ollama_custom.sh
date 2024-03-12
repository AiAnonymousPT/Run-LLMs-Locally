#!/bin/bash

# Step 1: Ensure the 'models' directory exists and download the model
mkdir -p ../models
wget -N -P ../models "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf"

# Step 2: Create a Makefile with the specified content
# echo "FROM models/mistral-7b-instruct-v0.2.Q4_K_M.gguf" > Modelfile

ollama create mistral-7b-Q4 -f Modelfile
ollama run mistral-7b-Q4