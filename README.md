# IoT-LLM: Enhancing Real-World IoT Task Reasoning with Large Language Models

## Framework Overview

![Framework Overview](./Images/overall_pipeline_final.png)

## Table of Contents

- [Key Features](#key-features)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Configuration](#configuration)
  - [API Keys (Closed-Source Models)](#1-api-keys-closed-source-models)
  - [Local Model Checkpoints (Open-Source Models)](#2-local-model-checkpoints-open-source-models)
  - [Embedding &amp; Ranking Models](#3-embedding--ranking-models)
- [Preparing Your Own Task](#preparing-your-own-task)
  - [Sensor Data](#1-sensor-data)
  - [Domain Knowledge](#2-domain-knowledge)
  - [Demo Knowledge (Few-Shot Examples)](#3-demo-knowledge-few-shot-examples)
- [Usage](#usage)
  - [Command-Line Arguments](#command-line-arguments)
  - [Running Experiments](#running-experiments)
- [How the Pipeline Works](#how-the-pipeline-works)

## Key Features

- **RAG-Enhanced Reasoning** — Retrieves domain knowledge and few-shot demonstrations to augment LLM prompts, significantly improving classification accuracy on IoT sensor data
- **Hybrid Retrieval** — Combines semantic embedding search with BM25 keyword search, followed by a cross-encoder re-ranker, for high-quality knowledge retrieval
- **Multi-Model Support** — Works with both closed-source API models (OpenAI GPT, Google Gemini, Anthropic Claude) and open-source local models (LLaMA2, Mistral) via the [Haystack](https://haystack.deepset.ai/) framework
- **Ablation-Ready** — Built-in flags (`--no_domain_knowledge`, `--no_demo_knowledge`) to easily disable individual knowledge sources for ablation studies
- **Modular & Extensible** — Clean separation between data loading, knowledge retrieval, prompt construction, and model inference, making it straightforward to adapt to new IoT tasks

## Directory Structure

```
IoT-Agent/
├── common/                        # Core framework modules
│   ├── args.py                    #   Command-line argument definitions
│   ├── data_calibration.py        #   Data calibration utilities
│   ├── generate_prompt.py         #   Prompt template construction
│   ├── model.py                   #   Model loading and inference wrappers
│   ├── read_data.py               #   Data loading and preprocessing
│   └── utils.py                   #   RAG pipeline utilities, evaluation, etc.
├── data/                          # [You provide] IoT sensor datasets
│   └── your_task_data/
├── knowledge/                     # [You provide] RAG knowledge base
│   └── your_task_type/
│       ├── domain-knowledge/      #     Expert knowledge documents
│       └── demo-knowledge/        #     Labeled few-shot examples
├── models/                        # [You provide] Local model checkpoints
│   └── your_model/
├── results/                       # Output logs and evaluation results
├── Images/                        # Framework overview images
├── main.py                        # Main entry point
├── imports.py                     # Central imports and global config
├── openAI_API_key.py              # [You create] API keys for closed-source models
├── run.sh                         # Shell script for running experiments
├── requirements.txt               # Python dependencies
└── README.md
```

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Morpheus-An/IoT-Agent.git
cd IoT-Agent
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate     # On Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> If you plan to run open-source models locally, make sure you have a CUDA-compatible GPU and the appropriate CUDA toolkit installed.

## Configuration

### 1. API Keys (Closed-Source Models)

To use closed-source LLMs, create a file named **`openAI_API_key.py`** in the project root with the following variables:

# openAI_API_key.py

# OpenAI (required for gpt3.5 / gpt4 / gpt4o-mini)
MY_API = "sk-your-openai-api-key"

# Google Gemini (required for gemini-* models)
GOOGLE_KEY = "your-google-api-key"

# Anthropic (required for claude)
CLAUDE_KEY = "sk-ant-your-anthropic-api-key"


You only need to fill in the keys for the models you actually plan to use.

> **Important**: Do **not** commit `openAI_API_key.py` to version control. Add it to your `.gitignore`.

**Available closed-source model keys**: `gpt3.5`, `gpt4`, `gpt4o-mini`, `gemini-pro`, `gemini-pro-flash`, `gemini-1.5-pro`, `gemini-1.5-pro-002`, `claude`

### 2. Local Model Checkpoints (Open-Source Models)

To use open-source models locally, download the model checkpoints and place them in the **`models/`** directory:

```
models/
├── LLaMa2-7b-32k/                # LLaMA2-7b-32k checkpoint
│   ├── config.json
│   ├── tokenizer.model
│   ├── model.safetensors          # or pytorch_model.bin
│   └── ...
└── Mistral-7b-instruct-v0.3/     # Mistral-7b checkpoint
    ├── config.json
    ├── tokenizer.model
    ├── model.safetensors
    └── ...
```

You can download them from Hugging Face:

```bash
pip install huggingface-hub

# LLaMA2-7b-32k
huggingface-cli download togethercomputer/LLaMA-2-7B-32K --local-dir models/LLaMa2-7b-32k

# Mistral-7b-Instruct-v0.3
huggingface-cli download mistralai/Mistral-7B-Instruct-v0.3 --local-dir models/Mistral-7b-instruct-v0.3
```

Then update the model paths in **`imports.py`** to point to your local directories:

```python
MODEL = {
    ...
    "llama2": "models/LLaMa2-7b-32k",
    "Mistral": "models/Mistral-7b-instruct-v0.3",
}
```

> Local models require a GPU with sufficient VRAM (at least 16 GB recommended for 7B models). Use `--device cuda` when running.

**Available open-source model keys**: `llama2`, `Mistral`

### 3. Embedding & Ranking Models

The RAG pipeline relies on two auxiliary models for retrieval. By default, they are loaded from local paths defined in `imports.py`. Update these paths to match your environment, or use the Hugging Face model IDs to download them automatically:

| Role          | Model             | Hugging Face ID                                                          |
| ------------- | ----------------- | ------------------------------------------------------------------------ |
| Text Embedder | GTE-Large         | [`thenlper/gte-large`](https://huggingface.co/thenlper/gte-large)         |
| Re-Ranker     | BGE-Reranker-Base | [`BAAI/bge-reranker-base`](https://huggingface.co/BAAI/bge-reranker-base) |

In **`imports.py`**, configure the paths:

```python
# Use Hugging Face IDs (auto-download) or local paths
EMBEDDER_MODEL_LOCAL = "thenlper/gte-large"          # or "/path/to/local/gte-large"
RANKER_MODEL_LOCAL   = "BAAI/bge-reranker-base"      # or "/path/to/local/bge-reranker-base"
```

## Preparing Your Own Task

Since this repository provides the **general retrieval and inference framework**, you need to prepare three components for your specific IoT task:

### 1. Sensor Data

Place your IoT sensor datasets in the **`data/`** directory, organized by task. The data loading logic resides in `common/read_data.py` — you will need to adapt or extend the data reading functions to handle your specific data format.

```
data/
└── your_task_name/
    ├── train_data.csv
    ├── test_data.csv
    └── ...
```

### 2. Domain Knowledge

Domain knowledge provides the LLM with expert-level context about your IoT task. Place your curated documents in the **`knowledge/{task_type}/domain-knowledge/`** directory.

```
knowledge/
└── your_task_type/
    └── domain-knowledge/
        ├── expert_guide.txt
        ├── signal_patterns.pdf
        └── sensor_specs.md
```

**Accepted file formats**: `.txt`, `.pdf`, `.md`

**What to include** — any expert knowledge that helps an LLM understand your sensor data and task:

- **Sensor descriptions**: What each sensor measures, units, sampling frequency, physical meaning of the signals
- **Domain expertise**: How to interpret signal patterns for classification (e.g., "high variance in acceleration typically indicates vigorous activity")
- **Class definitions**: Clear descriptions of what each class/label represents and its distinguishing characteristics
- **Textbook or paper excerpts**: Relevant technical background about the data domain

The retrieval pipeline will automatically chunk these documents, embed them, and retrieve the most relevant passages at inference time based on the query and input data.

### 3. Demo Knowledge (Few-Shot Examples)

Demo knowledge provides labeled sensor data examples that serve as few-shot demonstrations in the LLM prompt. Place them in the **`knowledge/{task_type}/demo-knowledge/`** directory.

```
knowledge/
└── your_task_type/
    └── demo-knowledge/
        ├── ClassA_1.txt
        ├── ClassA_2.txt
        ├── ClassB_1.txt
        ├── ClassB_2.txt
        └── ...
```

**File naming convention**: **`{label}_{index}.txt`**

The label portion of the filename is used as metadata for filtered retrieval — the pipeline retrieves examples matching the ground-truth class and contrastive (non-ground-truth) class separately to construct informative few-shot demonstrations.

**What each file should contain**: A representative sensor data sample (formatted as text) along with its ground truth label. For example:

```
1. Triaxial acceleration signal:
X-axis: 0.92g, 0.88g, 0.95g, ...
Y-axis: 0.12g, 0.15g, 0.11g, ...
Z-axis: 0.31g, 0.28g, 0.33g, ...
X-axis-mean=0.917g, X-axis-var=0.003
...
ANSWER: WALKING
```

> **Tip**: If the `demo-knowledge/` folder is empty, the system can auto-generate demo knowledge files from your dataset (see `write_demo_knowledge()` in `common/utils.py`). You can also start without demo knowledge by using the `--no_demo_knowledge` flag, and later curate high-quality examples for better results.

## Usage

### Command-Line Arguments

| Argument                  | Type  | Default                | Description                                                                                 |
| ------------------------- | ----- | ---------------------- | ------------------------------------------------------------------------------------------- |
| `--task_type`           | str   | `imu_HAR`            | Your IoT task identifier (must match folder names in `knowledge/` and data loading logic) |
| `--model`               | str   | `gpt4`               | Model to use (see available keys in[Configuration](#configuration))                            |
| `--cls_num`             | int   | `2`                  | Number of classes for classification                                                        |
| `--grd`                 | str   | `LAYING`             | Ground truth class label for the current evaluation                                         |
| `--sample_num`          | int   | `50`                 | Number of samples to evaluate                                                               |
| `--device`              | str   | `cuda`               | Compute device:`cuda` or `cpu`                                                          |
| `--temperature`         | float | `0.5`                | LLM generation temperature                                                                  |
| `--data_path`           | str   | `data`               | Path to sensor data directory                                                               |
| `--knowledge_path`      | str   | `knowledge`          | Path to knowledge base directory                                                            |
| `--output_file_path`    | str   | `results/output.log` | Path for output log file                                                                    |
| `--no_domain_knowledge` | flag  | `False`              | Disable domain knowledge retrieval                                                          |
| `--no_demo_knowledge`   | flag  | `False`              | Disable demo knowledge retrieval                                                            |
| `--debug`               | flag  | `False`              | Debug mode — prints the final prompt without running LLM inference                         |

### Running Experiments

**Using Python directly:**

```bash
python main.py \
    --task_type your_task_type \
    --model gemini-1.5-pro-002 \
    --cls_num 2 \
    --sample_num 50 \
    --grd YourClassLabel \
    --output_file_path results/your_experiment.log
```

**Using the shell script:**

```bash
# Usage: ./run.sh [model] [task_type] [sample_num] [output_file_path] [cls_num] [grd]
./run.sh gemini-1.5-pro-002 your_task_type 50 results/output.log 2 YourClassLabel
```

**Ablation experiments:**

```bash
# Without domain knowledge
python main.py --task_type your_task --model gpt4o-mini --no_domain_knowledge ...

# Without demo knowledge
python main.py --task_type your_task --model gpt4o-mini --no_demo_knowledge ...

# Without both (pure LLM baseline)
python main.py --task_type your_task --model gpt4o-mini --no_domain_knowledge --no_demo_knowledge ...
```

**Debug mode** (inspect the constructed prompt without calling the LLM):

```bash
python main.py --task_type your_task --model gpt4 --debug
```

## How the Pipeline Works

The framework constructs an augmented prompt for the LLM through the following stages:

```
Sensor Data ──────────────────────────────────────────────────┐
                                                              │
Domain Knowledge Documents                                    │
  └─→ Chunk & Embed ──→ Hybrid Retrieval ──→ Re-rank ──┐     │
        (Semantic + BM25)              (Cross-Encoder)  │     │
                                                        ├──→ Prompt Builder ──→ LLM ──→ Answer
Demo Knowledge Examples                                 │
  └─→ Chunk & Embed ──→ Filtered Retrieval ──→ Re-rank ┘
        (by class label: ground-truth & contrastive)
```

1. **Data Loading & Preprocessing** — Raw IoT sensor data is loaded and preprocessed (e.g., downsampling, feature extraction). Implemented in `common/read_data.py`.
2. **Domain Knowledge Retrieval** — Expert documents are chunked, embedded, and stored in an in-memory document store. At query time, a hybrid strategy combines:

   - **Semantic search** via sentence embeddings (`thenlper/gte-large`)
   - **Keyword search** via BM25

   Results from both retrievers are joined and re-ranked by a cross-encoder (`BAAI/bge-reranker-base`) to surface the most relevant passages.
3. **Demo Knowledge Retrieval** — Labeled examples are retrieved in two streams:

   - **Ground-truth examples** — filtered by the target class label
   - **Contrastive examples** — filtered by a different class label

   This provides the LLM with both positive and negative references for better discrimination.
4. **Prompt Construction** — A task-specific prompt is assembled from:

   - A role definition (expert persona for the task domain)
   - Retrieved domain knowledge context
   - Retrieved few-shot demonstration examples
   - The actual sensor data to classify
   - The classification question and response format instructions

   Prompt templates are defined in `common/generate_prompt.py`.
5. **LLM Inference** — The complete prompt is sent to the selected LLM. Model wrappers in `common/model.py` support OpenAI, Google Gemini, Anthropic Claude, and HuggingFace local models via the Haystack framework.
6. **Evaluation** — Model responses are logged to the output file and can be evaluated for classification accuracy using the built-in `eval_generated_ans()` function in `common/utils.py`.

## Citation

If you find this work useful in your research, please consider citing our paper.

```
@article{an2025iot,
  title={IoT-LLM: A framework for enhancing large language model reasoning from real-world sensor data},
  author={An, Tuo and Zhou, Yunjiao and Zou, Han and Yang, Jianfei},
  journal={Patterns},
  publisher={Elsevier}
}
```

## License

This project is released for academic and research purposes.
