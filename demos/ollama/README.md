## Olama

[Quickstart Docs](https://github.com/ollama/ollama/blob/main/README.md#quickstart)

### Custom Config Examle
Inspect the Makefile and ollam_custom.sh, then run `./ollama_custom.sh` to download and config 4bit quant Mistral-7b.

The model can now be referenced `ollama run mistral-7b-Q4`

### Run Server
`ollama serve`

Request specific model in the API request

`curl http://localhost:11434/api/generate -d '{
  "model": "mistral-7b-Q4",
  "prompt": "Why is the sky blue?"
}'`