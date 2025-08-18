# LLM Semantic Alignment

### Setup

Start by running `setup_env.sh`:

```
./setup_env.sh
```

You will then be prompted for a huggingface token and an OpenAI token. OpenAI token is unnecessary for most experiments, but you will need a huggingface token for inference.
The bash script will automatically install conda/pip dependencies and create an env `alignment_benchmark_env`

run:

```
conda init alignment_benchmark_env
```

And you're ready to run experiments
