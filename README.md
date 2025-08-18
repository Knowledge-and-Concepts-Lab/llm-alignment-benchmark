# LLM Semantic Alignment

### Setup

Start by running `setup_env.sh`:

```bash
./setup_env.sh
```

You will then be prompted for a huggingface token and an OpenAI token. OpenAI token is unnecessary for most experiments, but you will need a huggingface token for inference.
The bash script will automatically install conda/pip dependencies and create an env `alignment_benchmark_env`

run:

```bash
conda init alignment_benchmark_env
```

And you're ready to run experiments

### Run

All main experiment code is in `exp_battery.py`. The parameters for all experiments are determined by three JSON files in config:

**exp_battery.json**: experiment specific parameters, i.e number of embeddings dimensions, number of triplet items to run, and model instruction prompt.
**model_battery.json**: huggingface model and tokenizer path and other parameters, such as whether the model is instruct tuned
**stim_battery.json**: stimuli parameters, i.e column to access for concept strings, path for embeddings and items.

All experiments are run through exp_battery like so:

```
python3 script/exp_battery.py [EXPERIMENT] [MODEL] [STIMULUS] version_dir="version"
```
Where EXPERIMENT, MODEL, and STIMULUS must be json objects in the corresponding files. Version dir can be any str, for file organization and wandb tracking purposes.

i.e to run the main triplet experiment on a gemma model for all THINGS stimuli:

```
python3 script/exp_battery.py triplet_run_1_a gemma-3-12b-it THINGS version_dir=1a
```




