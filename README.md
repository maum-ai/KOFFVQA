# KOFFVQA

This is the official repository for the evaluation code of KOFFVQA.

* [Dataset](https://huggingface.co/datasets/maum-ai/KOFFVQA_Data)
* [Report](https://arxiv.org/abs/2503.23730)
* [Leaderboard](https://huggingface.co/spaces/maum-ai/KOFFVQA-Leaderboard)

## About KOFFVQA

KOFFVQA is a carefully crafted free-form visual question answering(VQA) benchmark in the Korean language consisting of 275 questions across 10 different tasks. Each question consists of an image, the corresponding question, and a list of objective scoring criteria for responses. For each given image-question pair, the response of a VLM is evaluated by an LLM judge that is simply instructed to score the response based on the given criteria. This allows our benchmark to utilize the free-form generation ability of VLMs without having to rely on subjective metrics for grading. In addition, the tasks in the benchmark are designed to cover as many aspects of VLM performance relevant to real-world applications as possible.

## How to Use

The evaluation process of KOFFVQA is in two stages: the generation of responses for each question using the VLM, and the grading of the responses by the judge LLM.

### Generation

`generate.py` uses the provided model to generate responses for each question in the benchmark, saving it to an output file in `.xlsx` format.

#### API model
```sh
API_KEY=YOUR_API_KEY python generate.py \
    [--data DATA_PATH] \
    --model MODEL_NAME \
    [--outfile OUTPUT_FILE_PATH] \
    [--verbose]
```

The following API models and their various available versions are supported:

* **OpenAI models:** gpt-4o, gpt-4o-mini, gpt-4-turbo
* **Gemini models:** gemini-1.5-flash, gemini-1.5-flash-8b, gemini-1.5-pro, gemini-1.0-pro
* **Claude models:** claude-3-5-sonnet, claude-3-5-haiku, claude-3-opus, claude-3-sonnet, claude-3-haiku

The names of the models as specified by their respective API formats can be used as the model name in the generation command. The code only checks if the model name contains the names of each model family, and uses the appropriate API.

#### Huggingface or local model
```sh
torchrun --nproc_per_node N generate.py \
    [--data DATA_PATH] \
    --model MODEL_NAME_OR_PATH \
    [--adapter ADAPTER_PATH] \
    [--outfile OUTPUT_FILE_PATH] \
    [--verbose]
```

If not an API model, the model name must be either the name of a model on Huggingface, or a path to a local directory containing the model in Huggingface format. Optionally, a path to a directory containing a LoRA adapter may be provided as well.

If there is not enough GPU memory to fit the entire model on one GPU, running `generate.py` with `python` instead of `torchrun` will automatically load the model across multiple GPU devices using Huggingface's `device_map='auto'`.

The `architectures` attribute of the configuration file of the model is used to determine how to load the model and use it for generation. The following values are supported:

* `MllamaForConditionalGeneration`
* `Qwen2VLForConditionalGeneration`
* `MiniCPMV`
* `InternVLChatModel`
* `Ovis`
* `InternLMXComposer2ForCausalLM`
* `LlavaNextForConditionalGeneration`
* `LlavaLlamaForCausalLM`\*
* `LlavaQwenForCausalLM`\*
* `LlavaForConditionalGeneration`\*\*
* `MolmoForCausalLM`

(\*Requires `pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git`)

(\*\*Only Pixtral models in Huggingface format are supported.)

Note that the model may not function properly even if it uses the above architectures, if the model is modified in certain ways.

Additionally, Pixtral models are supported, but due to their non-standard format, only locally downloaded models are supported. To use Pixtral models, `pip install mistral_inference` is required. (Community-made checkpoints that follow the usual Huggingface format can be used in the same way as other models, where the architecture will be `LlavaForConditionalGeneration`.)

#### Data and output file paths

The data path must point to the `.tsv` file containing the KOFFVQA benchmark. By default, this is `data/KOFFVQA.tsv`. If the argument is the default value and the file does not exist, it will automatically be downloaded from Huggingface.

The output file path is set to `result/{model name}_{adapter name}/{benchmark name}_gen.xlsx` by default.

### Grading

`evaluate.py` takes the output file of the generation process as input, and uses a local judge LLM to grade the responses using each question's grading criteria.

```sh
torchrun --nproc_per_node N evaluate.py \
    --predfile INPUT_FILE_PATH \
    [--outfile JUDGEMENT_FILE_PATH] \
    [--scorefile SCORE_FILE_PATH] \
    [--judge JUDGE_NAME] \
    [--verbose]
```

If there is not enough GPU memory to fit the entire judge on one GPU, running `evaluate.py` with `python` instead of `torchrun` will automatically load the judge across multiple GPU devices using Huggingface's `device_map='auto'`.

Currently, the only judge LLM supported is `google/gemma-2-9b-it`. This is due to the fact that this is the only available LLM that can both (1) reasonably fit on a local machine, and (2) understand the grading criteria given in Korean well enough to consistently grade generated responses in a way that aligns with human intention.

The results are saved in two files. The **judgement file**, in `.xlsx` format, contains the full response given by the judge LLM when grading the generated responses, for debugging purposes. The **score file**, in `.csv` format, contains the evaluation scores for each category and the overall score for the benchmark. By default, the file names for these are automatically generated from the input file, and saved to the same directory.

